from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from board_opt.builder import ToolchainPaths
from board_opt.eval_runner import run_full_eval, run_llama_bench
from board_opt.gate import evaluate_candidate
from board_opt.log import log_info, log_verbose
from board_opt.models import (
    BudgetCandidate,
    BudgetPlan,
    EvalMetrics,
    GateConfig,
    GGUFInfo,
    HardwareProfile,
)


@dataclass
class TunedRuntimeConfig:
    ctk: str
    ctv: str
    ubatch: int
    context: int
    flash_attn: bool
    cache_ram: int
    top_k: int
    env_vars: dict[str, str] = field(default_factory=dict)
    metrics: EvalMetrics | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TunedRuntimeConfig:
        d = dict(d)
        if isinstance(d.get("metrics"), dict):
            d["metrics"] = EvalMetrics.from_dict(d["metrics"])
        return cls(**d)


def tune_runtime(
    info: GGUFInfo,
    hw: HardwareProfile,
    plan: BudgetPlan,
    tc: ToolchainPaths | None,
    baseline: EvalMetrics,
    gate_cfg: GateConfig,
    domains: list[str],
    langs: list[str],
    calib_file: str = "",
    eval_file: str | None = None,
    runner_fn: Callable[[BudgetCandidate, bool], EvalMetrics] | None = None,
) -> TunedRuntimeConfig:
    top_k = 40 if (info.top_k == 0 and info.vocab_size > 50000) else (info.top_k or 40)
    env_vars: dict[str, str] = {}
    if hw.backend == "vulkan":
        env_vars["GGML_VK_MAX_NODES_PER_SUBMIT"] = "1024"

    if not plan.candidates:
        if plan.recommended_max_context <= 0:
            raise RuntimeError(
                f"Model weights ({plan.weights_mb} MiB) exceed usable VRAM ({plan.usable_vram_mb} MiB); "
                "cannot fit even with q4_0/q4_0 KV cache."
            )
        return TunedRuntimeConfig(
            ctk="q4_0",
            ctv="q4_0",
            ubatch=128,
            context=plan.recommended_max_context,
            flash_attn=True,
            cache_ram=0,
            top_k=top_k,
            env_vars=env_vars,
            metrics=baseline,
            history=[],
        )

    def _eval(cand: BudgetCandidate, quick: bool) -> EvalMetrics:
        if runner_fn is not None:
            return runner_fn(cand, quick)
        assert tc is not None
        return run_full_eval(
            tc=tc,
            gguf_path=info.path,
            calib_file=calib_file,
            domains=domains,
            langs=langs,
            ctk=cand.ctk,
            ctv=cand.ctv,
            ubatch=cand.ubatch,
            context=cand.context,
            top_k=top_k,
            vram_mb=cand.estimated_vram_mb,
            eval_file=eval_file,
            quick=quick,
            env_vars=env_vars,
        )

    # Establish apples-to-apples quick baseline on candidates[0] so quick=True subset
    # eval_total matches every candidate (prevents false rejections against full baseline)
    log_info(f"[Stage 4] Screening {len(plan.candidates)} KV/runtime candidates...")
    quick_base = _eval(plan.candidates[0], True)
    log_verbose(
        f"  [1/{len(plan.candidates)}] {plan.candidates[0].ctk}/{plan.candidates[0].ctv} ub={plan.candidates[0].ubatch}: "
        f"pp={quick_base.pp_tps:.1f} tg={quick_base.tg_tps:.1f} ppl={quick_base.ppl:.3f} eval={quick_base.eval_passed}/{quick_base.eval_total} (quick baseline)"
    )
    best_cand = plan.candidates[0]
    best_metrics = quick_base
    accepted_cands: list[tuple[BudgetCandidate, EvalMetrics]] = [(best_cand, quick_base)]
    history: list[dict[str, Any]] = []

    for idx, cand in enumerate(plan.candidates[1:], start=2):
        m = _eval(cand, True)
        # Compare against fixed quick_base (prevents cumulative PPL drift), then require beating best_metrics tg/vram
        dec_base = evaluate_candidate(quick_base, m, gate_cfg, vram_constrained=plan.vram_constrained)
        dec_best = evaluate_candidate(best_metrics, m, gate_cfg, vram_constrained=plan.vram_constrained)
        accepted = dec_base.accepted and dec_best.accepted
        shown_dec = dec_best if dec_base.accepted else dec_base
        log_verbose(
            f"  [{idx}/{len(plan.candidates)}] {cand.ctk}/{cand.ctv} ub={cand.ubatch}: "
            f"pp={m.pp_tps:.1f} tg={m.tg_tps:.1f} ppl={m.ppl:.3f} eval={m.eval_passed}/{m.eval_total} -> "
            f"{'KEEP' if accepted else 'DROP'} ({shown_dec.reason})"
        )
        history.append({
            "candidate": cand.to_dict(),
            "metrics": m.to_dict(),
            "decision": shown_dec.to_dict(),
        })
        if accepted:
            best_cand = cand
            best_metrics = m
            accepted_cands.append((cand, m))

    # Test Vulkan submit batching on winner when running live
    if runner_fn is None and tc is not None and hw.backend == "vulkan":
        _, tg_256 = run_llama_bench(tc, info.path, ctk=best_cand.ctk, ctv=best_cand.ctv, ubatch=best_cand.ubatch, env_vars={"GGML_VK_MAX_NODES_PER_SUBMIT": "256"})
        _, tg_1024 = run_llama_bench(tc, info.path, ctk=best_cand.ctk, ctv=best_cand.ctv, ubatch=best_cand.ubatch, env_vars={"GGML_VK_MAX_NODES_PER_SUBMIT": "1024"})
        env_vars["GGML_VK_MAX_NODES_PER_SUBMIT"] = "1024" if tg_1024 >= tg_256 else "256"

    # Full domain verification walking backwards through accepted candidates
    final_metrics = baseline
    chosen_cand = plan.candidates[0]
    for cand, _ in reversed(accepted_cands):
        if cand == plan.candidates[0]:
            chosen_cand = cand
            final_metrics = baseline
            break
        fm = _eval(cand, False)
        fdec = evaluate_candidate(baseline, fm, gate_cfg, vram_constrained=plan.vram_constrained)
        if fdec.accepted:
            chosen_cand = cand
            final_metrics = fm
            break
    best_cand = chosen_cand

    return TunedRuntimeConfig(
        ctk=best_cand.ctk,
        ctv=best_cand.ctv,
        ubatch=best_cand.ubatch,
        context=best_cand.context,
        flash_attn=True,
        cache_ram=0,
        top_k=top_k,
        env_vars=env_vars,
        metrics=final_metrics,
        history=history,
    )
