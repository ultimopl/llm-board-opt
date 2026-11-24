import json
import os
import struct
import pytest

from board_opt.gguf_io import GGML_TYPES, align_up, convert_tensor_bytes, inspect_gguf, write_kv_val, write_str
from board_opt.optimizer import _select_reap_experts, plan_requant_targets
from board_opt.tuner import tune_runtime
from board_opt.vocab_prune import B2U, prune_vocab_gguf
from board_opt.eval_runner import _extract_code_block, build_calibration_file, load_eval_tasks
from board_opt.deploy import check_active_gpu_procs, extract_smoke_reply
from board_opt.models import (
    BudgetCandidate,
    BudgetPlan,
    EvalMetrics,
    GateConfig,
    GGUFInfo,
    HardwareProfile,
    TensorInfo,
)
from board_opt.cli import validate_budget_or_abort


def test_ggml_types_nvfp4_and_q1_0_exact():
    assert GGML_TYPES[40] == (64, 36)   # NVFP4: QK=64, 36 bytes
    assert GGML_TYPES[41] == (128, 18)  # Q1_0: QK=128, 18 bytes


def test_convert_tensor_bytes_passes_ncols_then_nrows(monkeypatch, tmp_path):
    captured = {}

    def fake_call(cmd):
        captured["cmd"] = cmd
        # Write dummy output matching Q4_0 size for dims=[64, 4] -> 256 elems -> 8 blocks * 18 = 144 bytes
        with open(cmd[6], "wb") as f:
            f.write(b"\x00" * 144)

    fake_bin = tmp_path / "ggml_conv"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr("subprocess.check_call", fake_call)
    raw = b"\x00" * (4 * 36)  # 4 blocks of NVFP4 (dims=[64, 4])
    out = convert_tensor_bytes(raw, 40, 2, [64, 4], ggml_conv_bin=str(fake_bin))
    assert len(out) == 144
    # ggml_conv expects: <bin> <src_type> <dst_type> <ncols=dims[0]> <nrows=ne//dims[0]> <in> <out>
    assert captured["cmd"][3] == "64"
    assert captured["cmd"][4] == "4"


def test_select_reap_experts_reads_reap_trace_json(tmp_path):
    p = tmp_path / "reap.json"
    p.write_text(json.dumps({"n_tokens": 100, "reap": [[0.1, 0.9, 0.5, 0.2], [0.8, 0.1, 0.2, 0.7]]}))
    keep = _select_reap_experts(str(p), target_count=2)
    assert keep == {0: [1, 2], 1: [0, 3]}


def test_tuner_quick_subset_does_not_falsely_reject_against_full_baseline():
    hw = HardwareProfile("x86_64", "znver2", 6, 16384, 14746, "vulkan", "BC-250", True, True, False, False, False, ["Q4_0"])
    info = GGUFInfo("m.gguf", "llama", 8 * 1024**3, 32, 32, 8, 128, 128, 128000, "gpt2", 0, 0, 40, [], {})
    plan = BudgetPlan(
        usable_vram_mb=14746, weights_mb=8192, vram_constrained=False, recommended_max_context=32768,
        candidates=[
            BudgetCandidate("f16", "f16", 256, 32768, 12000),
            BudgetCandidate("q8_0", "q4_0", 256, 32768, 9800),
        ],
    )
    # Full baseline has 9 tasks (8/9), quick runner only runs 3 tasks (3/3)!
    base = EvalMetrics(300.0, 30.0, 6.30, 8, 9, 12000)

    def quick_aware_runner(cand, quick):
        if quick:
            if cand.ctk == "q8_0":
                return EvalMetrics(315.0, 34.0, 6.35, 3, 3, 9800)
            return EvalMetrics(300.0, 30.0, 6.30, 3, 3, 12000)
        if cand.ctk == "q8_0":
            return EvalMetrics(315.0, 34.0, 6.35, 8, 9, 9800)
        return base

    best = tune_runtime(info, hw, plan, None, base, GateConfig(), ["reasoning"], ["en"], runner_fn=quick_aware_runner)
    assert (best.ctk, best.ctv) == ("q8_0", "q4_0")


def test_vocab_prune_pads_token_list_when_not_multiple_of_64(tmp_path):
    # 256 byte tokens + 5 ASCII words + 100 CJK = 361 tokens -> kept = 261 (NOT multiple of 64 -> padded to 320!)
    tokens = [B2U[b] for b in range(256)] + ["hello", "world", "print", "return", "while"]
    for i in range(100):
        tokens.append("".join(B2U[b] for b in chr(0x4E00 + i).encode("utf-8")))
    n_vocab = len(tokens)
    dim = 32
    src = tmp_path / "unaligned.gguf"
    dst = tmp_path / "unaligned_pruned.gguf"

    kvs = [
        ("general.architecture", 8, "llama"),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens, n_vocab)),
        ("tokenizer.ggml.token_type", 9, (5, struct.pack(f"<{n_vocab}i", *([1] * n_vocab)), n_vocab)),
    ]
    with open(src, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)))
        for k, vt, val in kvs:
            write_str(f, k)
            f.write(struct.pack("<I", vt))
            write_kv_val(f, vt, val)
        write_str(f, "token_embd.weight")
        f.write(struct.pack("<IQQIQ", 2, dim, n_vocab, 0, 0))
        pad = align_up(f.tell(), 32) - f.tell()
        f.write(b"\x00" * pad)
        f.write(b"\x00" * (dim * n_vocab * 4))

    res = prune_vocab_gguf(str(src), str(dst), ["en"], min_vocab_to_prune=100)
    assert res["skipped"] is False
    assert res["padded_vocab"] == 320
    info = inspect_gguf(str(dst))
    # tokenizer.ggml.tokens length MUST match padded_vocab so llama.cpp loads it!
    assert info.vocab_size == 320
    assert info.tensors[0].dims == [dim, 320]


def test_validate_budget_aborts_when_model_cannot_fit():
    plan = BudgetPlan(usable_vram_mb=14000, weights_mb=15000, vram_constrained=True, recommended_max_context=0, candidates=[])
    with pytest.raises(RuntimeError, match="exceeds usable VRAM"):
        validate_budget_or_abort(plan, requested_ctx=32768)


def test_plan_requant_does_not_quantize_router_gate():
    hw = HardwareProfile("x86_64", "znver2", 6, 16384, 14746, "vulkan", "BC-250", True, True, False, False, False, ["Q4_0"])
    tensors = [
        TensorInfo("blk.0.ffn_gate_inp.weight", [2048, 128], 0, 1048576),   # MoE router gate -> KEEP F32!
        TensorInfo("blk.6.loop_0_ctrl_a.weight", [2048, 256], 0, 2097152),  # loop ctrl -> Q4_0
    ]
    targets = plan_requant_targets(tensors, hw)
    assert "blk.0.ffn_gate_inp.weight" not in targets
    assert targets["blk.6.loop_0_ctrl_a.weight"] == 2


def test_extract_code_block_does_not_match_cpp_as_c():
    text = "```cpp\nint main() { return 0; }\n```\n```c\nint f(void) { return 1; }\n```"
    assert _extract_code_block(text, "c") == "int f(void) { return 1; }"


def test_calibration_file_has_enough_tokens_and_3_pt_tasks(tmp_path):
    out = build_calibration_file(["en"], ["reasoning"], str(tmp_path / "cal.txt"))
    assert os.path.getsize(out) >= 32000
    pt_tasks = [t for t in load_eval_tasks([], ["pt"], quick=False) if t["domain"] == "pt"]
    assert len(pt_tasks) >= 3


def test_deploy_handles_null_content_and_ignores_own_pid():
    assert extract_smoke_reply({"choices": [{"message": {"content": None, "reasoning_content": "thinking"}}]}) == "thinking"
    # Own PID line must not trigger active GPU error
    check_active_gpu_procs([f"{os.getpid()} python3 board-opt --model llama-server.gguf"])
