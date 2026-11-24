import struct
from board_opt.inspector import summarize_layer_quants, summarize_vocab_scripts, format_inspection_report
from board_opt.gguf_io import align_up, write_kv_val, write_str
from board_opt.models import GGUFInfo, HardwareProfile, TensorInfo
from board_opt.quant_bench import QuantBenchEntry
from board_opt.vocab_prune import B2U


def test_summarize_layer_quants_computes_expected_savings_and_loss():
    hw = HardwareProfile("x86_64", "znver2", 6, 16384, 13909, "vulkan", "BC-250", True, True, False, False, False, ["Q4_0"])
    table = {
        "Q8_0": QuantBenchEntry("Q8_0", 8, True, 8.5, 0.10, 169.0, 180.0, 1200.0, 4.6, 0.008, 0.999987),
        "Q4_0": QuantBenchEntry("Q4_0", 2, True, 4.5, 0.08, 212.0, 119.0, 1278.0, 1.0, 0.138, 0.996358),
    }
    info = GGUFInfo(
        "m.gguf", "llama", 2 * 1024**3, 32, 32, 8, 128, 128, 128000, "gpt2", 128, 8, 40,
        tensors=[
            TensorInfo("blk.0.attn_q.weight", [4096, 4096], 8, 17825792),
            TensorInfo("blk.0.attn_k.weight", [4096, 4096], 8, 17825792),
        ],
        kv_meta={},
    )
    rows = summarize_layer_quants(info, hw, quant_table=table)
    assert len(rows) == 1
    assert rows[0]["role"] == "attn"
    assert rows[0]["src_type"] == "Q8_0"
    assert rows[0]["target_type"] == "Q4_0"
    assert rows[0]["est_vram_save_mb"] > 10.0
    assert rows[0]["est_tg_gain_pct"] > 20.0
    assert rows[0]["target_cos_sim"] == 0.996358


def test_summarize_vocab_scripts_counts_cjk_and_latin(tmp_path):
    pt_tok = "".join(B2U[b] for b in "coração".encode("utf-8"))
    tokens = [B2U[b] for b in range(256)] + ["hello", pt_tok]
    for i in range(50):
        tokens.append("".join(B2U[b] for b in chr(0x4E00 + i).encode("utf-8")))
    n_vocab = len(tokens)
    src = tmp_path / "v.gguf"
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

    summary = summarize_vocab_scripts(str(src))
    assert summary["prunable"] is True
    assert summary["scripts"]["cjk"]["count"] == 50
    assert summary["scripts"]["latin_ext"]["count"] >= 1
    report = format_inspection_report([], None, summary)
    assert "cjk" in report.lower()
