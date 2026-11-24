import pytest
from board_opt.gguf_io import inspect_gguf, tensor_nbytes
from board_opt.budget import kv_cache_bytes, solve_budget
from board_opt.models import GGUFInfo, HardwareProfile


def test_unknown_ggml_type_raises():
    with pytest.raises(ValueError, match="Unknown GGML type ID 999"):
        tensor_nbytes([32, 32], 999)


def test_kv_cache_bytes_q8_q4_vs_f16():
    info = GGUFInfo(
        path="m.gguf",
        arch="llama",
        file_size_bytes=8 * 1024**3,
        n_layer=32,
        n_layer_kv=32,
        n_head_kv=8,
        head_k=128,
        head_v=128,
        vocab_size=32000,
        tokenizer_model="gpt2",
        expert_count=0,
        expert_used_count=0,
        top_k=40,
        tensors=[],
        kv_meta={},
    )
    f16_b = kv_cache_bytes(info, 8192, "f16", "f16")
    q84_b = kv_cache_bytes(info, 8192, "q8_0", "q4_0")
    assert f16_b == 32 * 8192 * (8 * 128 * 2 + 8 * 128 * 2)
    assert q84_b < f16_b * 0.45


def test_solve_budget_oversized_model_never_ooms():
    hw = HardwareProfile(
        cpu_arch="x86_64",
        cpu_march="znver2",
        cpu_cores=6,
        ram_total_mb=16384,
        usable_vram_mb=14000,
        backend="vulkan",
        gpu_name="BC-250",
        uma=True,
        fp16=True,
        bf16=False,
        fp4=False,
        int_dot=False,
        preferred_formats=["Q4_0"],
    )
    huge = GGUFInfo(
        path="huge.gguf",
        arch="llama",
        file_size_bytes=15 * 1024**3,
        n_layer=40,
        n_layer_kv=40,
        n_head_kv=8,
        head_k=128,
        head_v=128,
        vocab_size=128000,
        tokenizer_model="gpt2",
        expert_count=0,
        expert_used_count=0,
        top_k=0,
        tensors=[],
        kv_meta={},
    )
    plan = solve_budget(huge, hw, requested_ctx=32768)
    assert plan.vram_constrained is True
    assert plan.candidates == []
    assert plan.recommended_max_context == 0
