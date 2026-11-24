from board_opt.quant_bench import QuantBenchEntry, pick_best_target_format
from board_opt.optimizer import plan_requant_targets
from board_opt.models import HardwareProfile, TensorInfo


def test_pick_best_target_format_uses_empirical_table_not_hardcode():
    table = {
        "Q8_0": QuantBenchEntry("Q8_0", 8, True, 8.5, 1.0, 120.0, 128.0, 800.0, 2.0, 0.001, 0.9999),
        "Q4_0": QuantBenchEntry("Q4_0", 2, True, 4.5, 0.6, 200.0, 112.0, 900.0, 1.5, 0.012, 0.9970),
        "Q4_K": QuantBenchEntry("Q4_K", 12, True, 4.5, 0.4, 300.0, 168.0, 1400.0, 4.0, 0.009, 0.9985),
        "IQ2_XXS": QuantBenchEntry("IQ2_XXS", 16, True, 2.06, 0.3, 400.0, 103.0, 500.0, 8.0, 0.090, 0.9400),
    }
    # On this board Q4_K is empirically faster than Q4_0 and passes cos_sim >= 0.995; IQ2_XXS fails cos_sim!
    best = pick_best_target_format(table, src_type=8, min_cos_sim=0.995, vram_constrained=False)
    assert best is not None
    assert best.name == "Q4_K"


def test_plan_requant_uses_empirical_quant_table():
    hw = HardwareProfile("x86_64", "znver2", 6, 16384, 14746, "vulkan", "SomeBoard", True, True, False, False, False, ["Q4_0"])
    table = {
        "Q8_0": QuantBenchEntry("Q8_0", 8, True, 8.5, 1.0, 120.0, 128.0, 800.0, 2.0, 0.001, 0.9999),
        "Q4_K": QuantBenchEntry("Q4_K", 12, True, 4.5, 0.4, 280.0, 157.0, 1300.0, 4.0, 0.009, 0.9985),
        "Q4_0": QuantBenchEntry("Q4_0", 2, True, 4.5, 0.6, 190.0, 107.0, 900.0, 1.5, 0.012, 0.9970),
    }
    tensors = [TensorInfo("blk.0.attn_q.weight", [4096, 4096], 8, 17825792)]
    targets = plan_requant_targets(tensors, hw, quant_table=table)
    assert targets == {"blk.0.attn_q.weight": 12}  # empirically picked Q4_K (type 12), not hardcoded Q4_0!
