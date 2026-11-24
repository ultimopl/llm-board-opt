from board_opt.tuner import tune_runtime
from board_opt.models import (
    BudgetCandidate,
    BudgetPlan,
    EvalMetrics,
    GateConfig,
    GGUFInfo,
    HardwareProfile,
)


def test_tuner_rejects_broken_kv_quant_and_picks_fastest_valid():
    hw = HardwareProfile(
        "x86_64", "znver2", 6, 16384, 14746, "vulkan", "BC-250",
        True, True, False, False, False, ["Q4_0"],
    )
    info = GGUFInfo(
        "m.gguf", "llama", 8 * 1024**3, 32, 32, 8, 128, 128,
        128000, "gpt2", 0, 0, 0, [], {},
    )
    plan = BudgetPlan(
        usable_vram_mb=14746,
        weights_mb=8192,
        vram_constrained=False,
        recommended_max_context=32768,
        candidates=[
            BudgetCandidate("f16", "f16", 256, 32768, 12000),
            BudgetCandidate("q8_0", "q4_0", 256, 32768, 9800),
            BudgetCandidate("q4_0", "q4_0", 256, 32768, 9200),
        ],
    )
    base = EvalMetrics(300.0, 30.0, 6.30, 10, 12, 12000)

    def fake_runner(cand, quick):
        if cand.ctk == "q4_0":
            return EvalMetrics(320.0, 38.0, 7.50, 5, 12, 9200)  # broken KV quant!
        if cand.ctk == "q8_0":
            return EvalMetrics(315.0, 34.0, 6.35, 10, 12, 9800)  # fast & valid
        return base

    best = tune_runtime(
        info, hw, plan, None, base, GateConfig(),
        ["reasoning"], ["en"], runner_fn=fake_runner,
    )
    assert (best.ctk, best.ctv) == ("q8_0", "q4_0")
    assert best.top_k == 40
