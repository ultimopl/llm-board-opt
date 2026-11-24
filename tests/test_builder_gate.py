from board_opt.builder import cmake_args_for_hardware
from board_opt.gate import StageStore, evaluate_candidate
from board_opt.models import EvalMetrics, GateConfig, HardwareProfile


def test_cmake_args_vulkan_native():
    hw = HardwareProfile(
        cpu_arch="x86_64",
        cpu_march="znver2",
        cpu_cores=6,
        ram_total_mb=16384,
        usable_vram_mb=14746,
        backend="vulkan",
        gpu_name="AMD BC-250",
        uma=True,
        fp16=True,
        bf16=False,
        fp4=False,
        int_dot=False,
        preferred_formats=["Q4_0"],
    )
    args = cmake_args_for_hardware(hw, "/src", "/build")
    assert "-DGGML_VULKAN=ON" in args
    assert "-DBUILD_SHARED_LIBS=OFF" in args
    assert "-DGGML_LTO=ON" in args


def test_gate_rejects_eval_regression_and_excess_ppl(tmp_path):
    base = EvalMetrics(pp_tps=300.0, tg_tps=30.0, ppl=6.30, eval_passed=10, eval_total=12, vram_mb=11000)
    cfg = GateConfig()  # allow_eval_drop=0, max_ppl_delta=0.03, min_tg_gain=0.015
    bad_eval = EvalMetrics(pp_tps=350.0, tg_tps=40.0, ppl=6.35, eval_passed=9, eval_total=12, vram_mb=10500)
    assert evaluate_candidate(base, bad_eval, cfg, vram_constrained=False).accepted is False
    bad_ppl = EvalMetrics(pp_tps=350.0, tg_tps=40.0, ppl=6.60, eval_passed=10, eval_total=12, vram_mb=10500)
    assert evaluate_candidate(base, bad_ppl, cfg, vram_constrained=False).accepted is False
    good_vram = EvalMetrics(pp_tps=300.0, tg_tps=30.1, ppl=6.35, eval_passed=10, eval_total=12, vram_mb=10700)
    assert evaluate_candidate(base, good_vram, cfg, vram_constrained=True).accepted is True
    assert evaluate_candidate(base, good_vram, cfg, vram_constrained=False).accepted is False

    store = StageStore(str(tmp_path))
    store.save("baseline", base.to_dict())
    assert store.load("baseline")["tg_tps"] == 30.0
