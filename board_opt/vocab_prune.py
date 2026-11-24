from __future__ import annotations

import os
import struct
from typing import Any

from board_opt.gguf_io import (
    align_up,
    read_gguf_header,
    tensor_nbytes,
    write_kv_val,
    write_str,
)

SUPPORTED_LANGS: set[str] = {"en", "pt", "es", "fr", "de", "it"}


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = gpt2_bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}
SINGLE_BYTE_TOKENS = set(B2U.values())


def build_allowed_charset(langs: list[str]) -> set[str]:
    allowed: set[str] = set()
    # Printable ASCII + whitespace/control (covers English + Python/C/Math syntax)
    for cp in range(0x00, 0x7F):
        allowed.add(chr(cp))
    # Common typographic punctuation & math symbols
    for cp in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x00B0, 0x00B1, 0x00D7, 0x00F7, 0x2264, 0x2265, 0x2260):
        allowed.add(chr(cp))

    norm_langs = {l.lower() for l in langs}
    if norm_langs & {"pt", "es", "fr", "de", "it"}:
        # Latin-1 Supplement (0x00A0..0x00FF: á, é, í, ó, ú, ã, õ, ç, ñ, ü, ß, etc.)
        for cp in range(0x00A0, 0x0100):
            allowed.add(chr(cp))
        # Latin Extended-A (0x0100..0x017F: œ, Œ, Ÿ, etc.)
        for cp in range(0x0100, 0x0180):
            allowed.add(chr(cp))
    return allowed


def _token_allowed(tok: str, allowed_chars: set[str]) -> bool:
    if tok in SINGLE_BYTE_TOKENS:
        return True
    try:
        raw = bytes(U2B[ch] for ch in tok)
    except KeyError:
        return False
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Keep only 2-byte UTF-8 prefixes (Latin-1/Ext-A starts with 0xC2..0xC5)
        return all(b < 0xE0 for b in raw)
    return all(ch in allowed_chars for ch in decoded)


def prune_vocab_gguf(
    src: str,
    dst: str,
    langs: list[str],
    min_vocab_to_prune: int = 64000,
) -> dict[str, Any]:
    ver, kvs, tensors, data_offset, alignment = read_gguf_header(src)
    kv_map = {k: (vt, val) for k, vt, val in kvs}

    tok_model = str(kv_map.get("tokenizer.ggml.model", (8, "unknown"))[1])
    if tok_model != "gpt2":
        return {
            "skipped": True,
            "reason": f"tokenizer model '{tok_model}' is not gpt2 byte-level BPE",
            "old_vocab": 0,
            "new_vocab": 0,
            "saved_bytes": 0,
        }

    if "tokenizer.ggml.tokens" not in kv_map:
        return {"skipped": True, "reason": "missing tokenizer.ggml.tokens", "old_vocab": 0, "new_vocab": 0, "saved_bytes": 0}

    tokens: list[str] = list(kv_map["tokenizer.ggml.tokens"][1][1])
    old_vocab = len(tokens)
    if old_vocab <= min_vocab_to_prune:
        return {
            "skipped": True,
            "reason": f"vocab {old_vocab} <= threshold {min_vocab_to_prune}",
            "old_vocab": old_vocab,
            "new_vocab": old_vocab,
            "saved_bytes": 0,
        }

    token_types: list[int] | None = None
    if "tokenizer.ggml.token_type" in kv_map:
        etype, raw_tt, cnt = kv_map["tokenizer.ggml.token_type"][1]
        if isinstance(raw_tt, (bytes, bytearray)):
            token_types = list(struct.unpack(f"<{cnt}i", raw_tt))
        else:
            token_types = [int(x) for x in raw_tt]

    merges: list[str] = []
    if "tokenizer.ggml.merges" in kv_map:
        merges = list(kv_map["tokenizer.ggml.merges"][1][1])

    allowed_chars = build_allowed_charset(langs)
    keep: set[int] = set()
    tok_to_id = {t: i for i, t in enumerate(tokens)}

    # Keep special/control tokens, byte-level tokens, and language-allowed tokens
    for i, t in enumerate(tokens):
        is_special = token_types is not None and token_types[i] != 1
        if is_special or _token_allowed(t, allowed_chars):
            keep.add(i)

    # Also keep any token referenced by tokenizer.ggml.*_token_id
    for k, (vt, val) in kv_map.items():
        if k.startswith("tokenizer.ggml.") and k.endswith("_token_id") and isinstance(val, int) and 0 <= val < old_vocab:
            keep.add(val)

    # Downward BPE merge closure so every kept token remains 100% reachable
    merge_parents: dict[str, tuple[str, str]] = {}
    for m in merges:
        parts = m.split(" ", 1)
        if len(parts) == 2:
            merged = parts[0] + parts[1]
            if merged not in merge_parents:
                merge_parents[merged] = (parts[0], parts[1])

    stack = [tokens[i] for i in keep]
    while stack:
        cur = stack.pop()
        if cur in merge_parents:
            a, b = merge_parents[cur]
            for p in (a, b):
                pid = tok_to_id.get(p)
                if pid is not None and pid not in keep:
                    keep.add(pid)
                    stack.append(p)

    kept_ids = sorted(keep)
    new_vocab = len(kept_ids)
    if new_vocab >= int(old_vocab * 0.95):
        return {
            "skipped": True,
            "reason": f"pruning saves <5% ({old_vocab} -> {new_vocab})",
            "old_vocab": old_vocab,
            "new_vocab": new_vocab,
            "saved_bytes": 0,
        }

    old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_ids)}
    new_tokens = [tokens[i] for i in kept_ids]
    kept_tok_set = set(new_tokens)
    new_merges = [
        m for m in merges
        if len(m.split(" ", 1)) == 2
        and m.split(" ", 1)[0] in kept_tok_set
        and m.split(" ", 1)[1] in kept_tok_set
        and (m.split(" ", 1)[0] + m.split(" ", 1)[1]) in kept_tok_set
    ]

    # Pad embedding rows to multiple of 64 for clean GPU alignment
    padded_vocab = align_up(new_vocab, 64)
    pad_count = padded_vocab - new_vocab

    new_kvs: list[tuple[str, int, Any]] = []
    for k, vt, val in kvs:
        if k == "tokenizer.ggml.tokens":
            new_kvs.append((k, 9, (8, new_tokens, new_vocab)))
        elif k == "tokenizer.ggml.merges":
            new_kvs.append((k, 9, (8, new_merges, len(new_merges))))
        elif k == "tokenizer.ggml.token_type" and token_types is not None:
            ntt = struct.pack(f"<{new_vocab}i", *(token_types[i] for i in kept_ids))
            new_kvs.append((k, 9, (5, ntt, new_vocab)))
        elif k == "tokenizer.ggml.scores" and vt == 9:
            etype, raw_sc, cnt = val
            sc = struct.unpack(f"<{cnt}f", raw_sc)
            nsc = struct.pack(f"<{new_vocab}f", *(sc[i] for i in kept_ids))
            new_kvs.append((k, 9, (6, nsc, new_vocab)))
        elif k.startswith("tokenizer.ggml.") and k.endswith("_token_id") and isinstance(val, int) and val in old_to_new:
            new_kvs.append((k, vt, old_to_new[val]))
        elif k.endswith(".vocab_size") and isinstance(val, int):
            new_kvs.append((k, vt, padded_vocab))
        else:
            new_kvs.append((k, vt, val))

    # Plan tensor slicing (supports both separate and tied embeddings!)
    embed_names = {"token_embd.weight", "output.weight"}
    out_tensors: list[dict[str, Any]] = []
    cur_off = 0
    saved_bytes = 0
    for t in tensors:
        name = t["name"]
        dims = list(t["dims"])
        ttype = t["type"]
        src_nb = tensor_nbytes(dims, ttype)
        is_emb = name in embed_names and len(dims) == 2 and dims[1] >= old_vocab
        if is_emb:
            dims[1] = padded_vocab
        dst_nb = tensor_nbytes(dims, ttype)
        saved_bytes += max(0, src_nb - dst_nb)
        out_tensors.append({
            "name": name,
            "src_dims": t["dims"],
            "dims": dims,
            "type": ttype,
            "src_offset": t["offset"],
            "offset": cur_off,
            "src_nbytes": src_nb,
            "nbytes": dst_nb,
            "is_emb": is_emb,
        })
        cur_off = align_up(cur_off + dst_nb, alignment)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(b"GGUF")
        fout.write(struct.pack("<IQQ", ver, len(out_tensors), len(new_kvs)))
        for k, vt, val in new_kvs:
            write_str(fout, k)
            fout.write(struct.pack("<I", vt))
            write_kv_val(fout, vt, val)
        for t in out_tensors:
            write_str(fout, t["name"])
            fout.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                fout.write(struct.pack("<Q", d))
            fout.write(struct.pack("<IQ", t["type"], t["offset"]))
        pad = align_up(fout.tell(), alignment) - fout.tell()
        if pad:
            fout.write(b"\x00" * pad)
        out_data_start = fout.tell()

        for t in out_tensors:
            fin.seek(data_offset + t["src_offset"])
            fout.seek(out_data_start + t["offset"])
            if not t["is_emb"]:
                remaining = t["src_nbytes"]
                while remaining > 0:
                    chunk = fin.read(min(remaining, 16 * 1024 * 1024))
                    fout.write(chunk)
                    remaining -= len(chunk)
            else:
                row_bytes = t["src_nbytes"] // t["src_dims"][1]
                raw = fin.read(t["src_nbytes"])
                for i in kept_ids:
                    fout.write(raw[i * row_bytes : (i + 1) * row_bytes])
                if pad_count > 0:
                    fout.write(b"\x00" * (pad_count * row_bytes))
        final_pad = align_up(fout.tell(), alignment) - fout.tell()
        if final_pad:
            fout.write(b"\x00" * final_pad)

    return {
        "skipped": False,
        "reason": "pruned",
        "old_vocab": old_vocab,
        "new_vocab": new_vocab,
        "padded_vocab": padded_vocab,
        "saved_bytes": saved_bytes,
    }
