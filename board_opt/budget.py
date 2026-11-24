from __future__ import annotations

from board_opt.models import BudgetCandidate, BudgetPlan, GGUFInfo, HardwareProfile

# Bytes per element for KV cache quantization types
KV_BPE: dict[str, float] = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
}

KV_PAIRS: list[tuple[str, str]] = [
    ("f16", "f16"),
    ("q8_0", "q8_0"),
    ("q8_0", "q4_0"),
    ("q4_0", "q4_0"),
]

UBATCH_SIZES: list[int] = [256, 512, 128]


def kv_cache_bytes(info: GGUFInfo, ctx: int, ctk: str, ctv: str) -> int:
    if ctk not in KV_BPE or ctv not in KV_BPE:
        raise ValueError(f"Unsupported KV cache type: ({ctk}, {ctv})")
    k_per_tok = info.n_head_kv * info.head_k * KV_BPE[ctk]
    v_per_tok = info.n_head_kv * info.head_v * KV_BPE[ctv]
    return int(round(info.n_layer_kv * ctx * (k_per_tok + v_per_tok)))


def compute_scratch_mb(ubatch: int) -> int:
    # Conservative Vulkan compute buffer + runtime overhead estimate
    return 160 + (ubatch // 2)


def solve_budget(
    info: GGUFInfo,
    hw: HardwareProfile,
    requested_ctx: int,
    target_vram_mb: int | None = None,
) -> BudgetPlan:
    usable_mb = target_vram_mb if target_vram_mb is not None else hw.usable_vram_mb
    weights_mb = (info.file_size_bytes + 1024 * 1024 - 1) // (1024 * 1024)

    f16_kv_mb = (kv_cache_bytes(info, requested_ctx, "f16", "f16") + 1024 * 1024 - 1) // (1024 * 1024)
    vram_constrained = (weights_mb + f16_kv_mb + compute_scratch_mb(256)) > usable_mb or (target_vram_mb is not None)

    candidates: list[BudgetCandidate] = []
    for ctk, ctv in KV_PAIRS:
        kv_mb = (kv_cache_bytes(info, requested_ctx, ctk, ctv) + 1024 * 1024 - 1) // (1024 * 1024)
        for ub in UBATCH_SIZES:
            est = weights_mb + kv_mb + compute_scratch_mb(ub)
            if est <= usable_mb:
                candidates.append(
                    BudgetCandidate(
                        ctk=ctk,
                        ctv=ctv,
                        ubatch=ub,
                        context=requested_ctx,
                        estimated_vram_mb=est,
                    )
                )

    # Recommend max safe context using q4_0/q4_0 KV if requested_ctx doesn't fit
    rem_mb = usable_mb - weights_mb - compute_scratch_mb(128)
    if rem_mb <= 0:
        rec_ctx = 0
    else:
        per_tok_q4 = kv_cache_bytes(info, 1, "q4_0", "q4_0")
        if per_tok_q4 <= 0:
            rec_ctx = requested_ctx
        else:
            rec_ctx = int((rem_mb * 1024 * 1024) // per_tok_q4)
            rec_ctx = (rec_ctx // 1024) * 1024

    return BudgetPlan(
        usable_vram_mb=usable_mb,
        weights_mb=weights_mb,
        vram_constrained=vram_constrained,
        recommended_max_context=rec_ctx,
        candidates=candidates,
    )
