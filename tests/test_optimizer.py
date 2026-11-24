from board_opt.optimizer import plan_requant_targets, plan_moe_expert_steps
from board_opt.models import GGUFInfo, HardwareProfile, TensorInfo
from board_opt.recovery import NoOpRecovery


def test_plan_requant_targets_bc250():
    hw = HardwareProfile(
        "x86_64", "znver2", 6, 16384, 14746, "vulkan", "BC-250",
        True, True, False, False, False, ["Q4_0"],
    )
    tensors = [
        TensorInfo("blk.0.attn_q.weight", [4096, 4096], 8, 17825792),
        TensorInfo("blk.0.ffn_down_exps.weight", [4096, 4096, 8], 40, 100),
        TensorInfo("blk.0.loop_ctrl.weight", [1024, 1024], 0, 4194304),
        TensorInfo("blk.0.attn_norm.weight", [4096], 0, 16384),
    ]
    targets = plan_requant_targets(tensors, hw)
    assert targets == {
        "blk.0.attn_q.weight": 2,
        "blk.0.ffn_down_exps.weight": 2,
        "blk.0.loop_ctrl.weight": 2,
    }


def test_plan_moe_expert_steps():
    info = GGUFInfo(
        "moe.gguf", "inkling2_loop", 10 * 1024**3, 32, 32, 8, 128, 128,
        128000, "gpt2", 128, 8, 40, [], {},
    )
    steps = plan_moe_expert_steps(info, vram_constrained=True, prune_experts=96)
    assert steps["active_candidates"] == [6, 4]
    assert steps["reap_target"] == 96


def test_noop_recovery_returns_same_path():
    assert NoOpRecovery().run("/tmp/m.gguf", None, None) == "/tmp/m.gguf"
