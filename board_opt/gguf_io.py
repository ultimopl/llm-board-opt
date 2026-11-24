from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np

from board_opt.models import GGUFInfo, TensorInfo

# GGML tensor types: (block_size, type_size_bytes)
GGML_TYPES: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),   # MXFP4
    40: (16, 9),    # NVFP4
    41: (32, 6),    # Q1_0
    42: (64, 18),   # Q2_0
}

TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}

NAME_TO_TYPE: dict[str, int] = {v: k for k, v in TYPE_NAMES.items()}

SCALAR_FMT: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def tensor_nbytes(dims: list[int], ttype: int) -> int:
    if ttype not in GGML_TYPES:
        raise ValueError(f"Unknown GGML type ID {ttype} for dims={dims}")
    blck, tsz = GGML_TYPES[ttype]
    ne = 1
    for d in dims:
        ne *= d
    if ne % blck != 0:
        raise ValueError(f"Tensor ne={ne} not divisible by block size {blck} for type {ttype}")
    return (ne // blck) * tsz


def read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="surrogateescape")


def write_str(f, s: str) -> None:
    b = s.encode("utf-8", errors="surrogateescape")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def read_kv_val(f, vtype: int) -> Any:
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        etype, cnt = struct.unpack("<IQ", f.read(12))
        if etype in SCALAR_FMT:
            sz = struct.calcsize(SCALAR_FMT[etype])
            return (etype, f.read(sz * cnt), cnt)
        return (etype, [read_kv_val(f, etype) for _ in range(cnt)], cnt)
    fmt = SCALAR_FMT[vtype]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def write_kv_val(f, vtype: int, val: Any) -> None:
    if vtype == 8:
        write_str(f, val)
    elif vtype == 9:
        etype, data, cnt = val
        f.write(struct.pack("<IQ", etype, cnt))
        if etype in SCALAR_FMT and isinstance(data, (bytes, bytearray)):
            f.write(data)
        else:
            for item in data:
                write_kv_val(f, etype, item)
    else:
        f.write(struct.pack(SCALAR_FMT[vtype], val))


def read_gguf_header(path: str) -> tuple[int, list[tuple[str, int, Any]], list[dict[str, Any]], int, int]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        kvs: list[tuple[str, int, Any]] = []
        alignment = 32
        for _ in range(n_kv):
            k = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            val = read_kv_val(f, vt)
            if k == "general.alignment":
                alignment = int(val)
            kvs.append((k, vt, val))
        tensors: list[dict[str, Any]] = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data_offset = align_up(f.tell(), alignment)
    return ver, kvs, tensors, data_offset, alignment


def inspect_gguf(path: str) -> GGUFInfo:
    _, kvs, raw_tensors, _, _ = read_gguf_header(path)
    kv_dict: dict[str, Any] = {}
    vocab_size = 0
    for k, vt, val in kvs:
        if k == "tokenizer.ggml.tokens" and vt == 9:
            vocab_size = int(val[2])
        elif vt != 9:
            kv_dict[k] = val

    arch = str(kv_dict.get("general.architecture", "unknown"))
    n_layer = int(kv_dict.get(f"{arch}.block_count", 0))
    n_head = int(kv_dict.get(f"{arch}.attention.head_count", 1))
    n_head_kv_raw = kv_dict.get(f"{arch}.attention.head_count_kv", n_head)
    n_head_kv = int(n_head_kv_raw) if isinstance(n_head_kv_raw, (int, float)) else int(n_head)
    n_embd = int(kv_dict.get(f"{arch}.embedding_length", 0))
    default_head = (n_embd // n_head) if n_head > 0 else 128
    head_k = int(kv_dict.get(f"{arch}.attention.key_length", default_head))
    head_v = int(kv_dict.get(f"{arch}.attention.value_length", head_k))

    # Account for hybrid/recurrent models (e.g. qwen35 full_attention_interval)
    interval = int(kv_dict.get(f"{arch}.full_attention_interval", 0))
    if interval > 1 and n_layer > 0:
        n_layer_kv = n_layer // interval
    else:
        n_layer_kv = n_layer

    expert_count = int(kv_dict.get(f"{arch}.expert_count", 0))
    expert_used_count = int(kv_dict.get(f"{arch}.expert_used_count", 0))
    tokenizer_model = str(kv_dict.get("tokenizer.ggml.model", "unknown"))
    top_k = int(kv_dict.get("general.sampling.top_k", 40))

    tensors = [
        TensorInfo(name=t["name"], dims=t["dims"], ttype=t["type"], nbytes=tensor_nbytes(t["dims"], t["type"]))
        for t in raw_tensors
    ]
    return GGUFInfo(
        path=path,
        arch=arch,
        file_size_bytes=os.path.getsize(path),
        n_layer=n_layer,
        n_layer_kv=n_layer_kv,
        n_head_kv=n_head_kv,
        head_k=head_k,
        head_v=head_v,
        vocab_size=vocab_size,
        tokenizer_model=tokenizer_model,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        top_k=top_k,
        tensors=tensors,
        kv_meta=kv_dict,
    )


def dequant_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    nb = n_elements // 32
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = arr[:, :2].view(np.float16).astype(np.float32)
    qs = arr[:, 2:].view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)


def quant_q4_0(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    idx = np.argmax(np.abs(x), axis=1)
    maxv = x[np.arange(x.shape[0]), idx]
    d = maxv / -8.0
    id_val = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
    q = np.clip(np.trunc(x * id_val[:, None] + 8.5), 0, 15).astype(np.uint8)
    packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def convert_tensor_bytes(
    raw: bytes,
    src_type: int,
    dst_type: int,
    dims: list[int],
    ggml_conv_bin: str | None = None,
) -> bytes:
    if src_type == dst_type:
        return raw
    ne = 1
    for d in dims:
        ne *= d
    if src_type == 0 and dst_type == 1:
        return np.frombuffer(raw, dtype=np.float32).astype(np.float16).tobytes()
    if src_type == 0 and dst_type == 2:
        return quant_q4_0(np.frombuffer(raw, dtype=np.float32))
    if src_type == 8 and dst_type == 2:
        return quant_q4_0(dequant_q8_0(raw, ne))
    if ggml_conv_bin and os.path.exists(ggml_conv_bin):
        nrows = ne // dims[0]
        with tempfile.NamedTemporaryFile(delete=False) as fin, tempfile.NamedTemporaryFile(delete=False) as fout:
            fin_p, fout_p = fin.name, fout.name
            fin.write(raw)
        try:
            subprocess.check_call(
                [ggml_conv_bin, str(src_type), str(dst_type), str(nrows), str(dims[0]), fin_p, fout_p]
            )
            with open(fout_p, "rb") as f:
                return f.read()
        finally:
            for p in (fin_p, fout_p):
                if os.path.exists(p):
                    os.unlink(p)
    raise RuntimeError(f"Unsupported tensor conversion {src_type} -> {dst_type} without ggml_conv")


def edit_gguf(
    src: str,
    dst: str,
    kv_overrides: dict[str, tuple[int, Any]] | None = None,
    tensor_requant: dict[str, int] | None = None,
    expert_keep: dict[int, list[int]] | None = None,
    ggml_conv_bin: str | None = None,
) -> None:
    kv_overrides = kv_overrides or {}
    tensor_requant = tensor_requant or {}
    ver, kvs, tensors, data_offset, alignment = read_gguf_header(src)

    new_kvs: list[tuple[str, int, Any]] = []
    seen: set[str] = set()
    for k, vt, val in kvs:
        if k in kv_overrides:
            ovt, oval = kv_overrides[k]
            new_kvs.append((k, ovt, oval))
            seen.add(k)
        else:
            new_kvs.append((k, vt, val))
    for k, (ovt, oval) in kv_overrides.items():
        if k not in seen:
            new_kvs.append((k, ovt, oval))

    # Plan output tensor metadata
    out_tensors: list[dict[str, Any]] = []
    cur_off = 0
    for t in tensors:
        name = t["name"]
        dims = list(t["dims"])
        ttype = t["type"]
        keep_idx = None
        if expert_keep:
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx in expert_keep:
                    if name.endswith("_exps.weight") and len(dims) == 3:
                        keep_idx = expert_keep[layer_idx]
                        dims[2] = len(keep_idx)
                    elif name.endswith("ffn_gate_inp.weight") and len(dims) == 2:
                        keep_idx = expert_keep[layer_idx]
                        dims[1] = len(keep_idx)
        dst_type = tensor_requant.get(name, ttype)
        nb = tensor_nbytes(dims, dst_type)
        out_tensors.append({
            "name": name,
            "src_dims": t["dims"],
            "dims": dims,
            "src_type": ttype,
            "type": dst_type,
            "src_offset": t["offset"],
            "offset": cur_off,
            "src_nbytes": tensor_nbytes(t["dims"], ttype),
            "nbytes": nb,
            "keep_idx": keep_idx,
        })
        cur_off = align_up(cur_off + nb, alignment)

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
            raw = fin.read(t["src_nbytes"])
            if t["keep_idx"] is not None:
                n_orig = t["src_dims"][-1]
                slice_bytes = t["src_nbytes"] // n_orig
                raw = b"".join(raw[i * slice_bytes : (i + 1) * slice_bytes] for i in t["keep_idx"])
            if t["type"] != t["src_type"]:
                raw = convert_tensor_bytes(raw, t["src_type"], t["type"], t["dims"], ggml_conv_bin)
            if len(raw) != t["nbytes"]:
                raise RuntimeError(f"Size mismatch for {t['name']}: got {len(raw)}, expected {t['nbytes']}")
            fout.seek(out_data_start + t["offset"])
            fout.write(raw)
        final_pad = align_up(fout.tell(), alignment) - fout.tell()
        if final_pad:
            fout.write(b"\x00" * final_pad)
