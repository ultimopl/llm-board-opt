import struct
from board_opt.vocab_prune import prune_vocab_gguf, build_allowed_charset


def test_allowed_charset_supports_en_pt_es_fr_de_it():
    chars = build_allowed_charset(["en", "pt", "de"])
    for ch in ["ç", "ã", "é", "ü", "ß", "{", "}"]:
        assert ch in chars
    assert "中" not in chars


def test_prune_skips_sentencepiece(tmp_path):
    src = tmp_path / "sp.gguf"
    buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
    k = b"tokenizer.ggml.model"
    v = b"llama"
    buf += struct.pack("<Q", len(k)) + k + struct.pack("<IQ", 8, len(v)) + v
    src.write_bytes(buf)
    res = prune_vocab_gguf(str(src), str(tmp_path / "out.gguf"), ["en", "pt"])
    assert res["skipped"] is True
    assert "gpt2" in res["reason"].lower()


def test_prune_tied_embeddings_roundtrip(tmp_path):
    from board_opt.gguf_io import align_up, inspect_gguf, write_kv_val, write_str
    from board_opt.vocab_prune import B2U

    # Build minimal GPT-2 BPE GGUF with tied embeddings (only token_embd.weight, no output.weight)
    tokens = [B2U[b] for b in range(256)]
    # Add 200 CJK tokens that should be pruned for ["en", "pt"]
    for i in range(200):
        cjk_bytes = chr(0x4E00 + i).encode("utf-8")
        tokens.append("".join(B2U[b] for b in cjk_bytes))
    n_vocab = len(tokens)  # 456
    dim = 32
    src = tmp_path / "tied.gguf"
    dst = tmp_path / "tied_pruned.gguf"

    kvs = [
        ("general.architecture", 8, "llama"),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens, n_vocab)),
        ("tokenizer.ggml.eos_token_id", 4, 255),
    ]
    with open(src, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)))
        for k, vt, val in kvs:
            write_str(f, k)
            f.write(struct.pack("<I", vt))
            write_kv_val(f, vt, val)
        write_str(f, "token_embd.weight")
        f.write(struct.pack("<IQQIQ", 2, dim, n_vocab, 0, 0))  # F32
        pad = align_up(f.tell(), 32) - f.tell()
        f.write(b"\x00" * pad)
        f.write(b"\x00" * (dim * n_vocab * 4))

    res = prune_vocab_gguf(str(src), str(dst), ["en", "pt"], min_vocab_to_prune=100)
    assert res["skipped"] is False
    assert res["new_vocab"] == 256
    assert res["saved_bytes"] > 0
    info = inspect_gguf(str(dst))
    assert info.vocab_size == 256
    assert len(info.tensors) == 1
    assert info.tensors[0].dims == [dim, 256]


def test_prune_cut_scripts_selective_only_cuts_chosen_scripts(tmp_path):
    from board_opt.gguf_io import align_up, inspect_gguf, write_kv_val, write_str
    from board_opt.vocab_prune import B2U

    tokens = [B2U[b] for b in range(256)]
    # Add 100 CJK + 50 Cyrillic tokens; cut ONLY cjk -> Cyrillic must survive!
    for i in range(100):
        tokens.append("".join(B2U[b] for b in chr(0x4E00 + i).encode("utf-8")))
    for i in range(50):
        tokens.append("".join(B2U[b] for b in chr(0x0410 + (i % 32)).encode("utf-8")))
    n_vocab = len(tokens)
    src = tmp_path / "sel.gguf"
    dst = tmp_path / "sel_pruned.gguf"
    kvs = [
        ("general.architecture", 8, "llama"),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens, n_vocab)),
    ]
    with open(src, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)))
        for k, vt, val in kvs:
            write_str(f, k)
            f.write(struct.pack("<I", vt))
            write_kv_val(f, vt, val)
        write_str(f, "token_embd.weight")
        f.write(struct.pack("<IQQIQ", 2, 32, n_vocab, 0, 0))
        pad = align_up(f.tell(), 32) - f.tell()
        f.write(b"\x00" * pad)
        f.write(b"\x00" * (32 * n_vocab * 4))

    res = prune_vocab_gguf(str(src), str(dst), [], min_vocab_to_prune=100, cut_scripts=["cjk"])
    assert res["skipped"] is False
    assert res["new_vocab"] == 306  # 256 bytes + 50 cyrillic kept; 100 cjk cut!
    assert inspect_gguf(str(dst)).vocab_size == 320  # padded to multiple of 64
