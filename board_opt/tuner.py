from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from board_opt.builder import ToolchainPaths
from board_opt.eval_runner import run_full_eval, run_llama_bench
from board_opt.gate import evaluate_candidate
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
        return TunedRuntimeConfig(
            ctk="q8_0",
            ctv="q4_0",
            ubatch=128,
            context=max(2048, plan.recommended_max_context),
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

    best_cand = plan.candidates[0]
    best_metrics = baseline
    history: list[dict[str, Any]] = []

    for cand in plan.candidates:
        m = _eval(cand, True)
        dec = evaluate_candidate(best_metrics, m, gate_cfg, vram_constrained=plan.vram_constrained)
        history.append({
            "candidate": cand.to_dict(),
            "metrics": m.to_dict(),
            "decision": dec.to_dict(),
        })
        if dec.accepted:
            best_cand = cand
            best_metrics = m

    # Test Vulkan submit batching on winner when running live
    if runner_fn is None and tc is not None and hw.backend == "vulkan":
        _, tg_256 = run_llama_bench(tc, info.path, ctk=best_cand.ctk, ctv=best_cand.ctv, ubatch=best_cand.ubatch, env_vars={"GGML_VK_MAX_NODES_PER_SUBMIT": "256"})
        _, tg_1024 = run_llama_bench(tc, info.path, ctk=best_cand.ctk, ctv=best_cand.ctv, ubatch=best_cand.ubatch, env_vars={"GGML_VK_MAX_NODES_PER_SUBMIT": "1024"})
        env_vars["GGML_VK_MAX_NODES_PER_SUBMIT"] = "1024" if tg_1024 >= tg_256 else "256"

    # Full domain verification on the winning candidate
    final_metrics = _eval(best_cand, False)
    final_dec = evaluate_candidate(baseline, final_metrics, gate_cfg, vram_constrained=plan.vram_constrained)
    if not final_dec.accepted and best_cand != plan.candidates[0]:
        # Fall back to first safe candidate if full eval caught a regression
        best_cand = plan.candidates[0]
        final_metrics = baseline

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
