from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from board_opt.builder import ToolchainPaths
from board_opt.eval_runner import run_full_eval
from board_opt.gate import evaluate_candidate
from board_opt.gguf_io import NAME_TO_TYPE, edit_gguf, inspect_gguf
from board_opt.models import EvalMetrics, GateConfig, GGUFInfo, HardwareProfile, TensorInfo
from board_opt.recovery import NoOpRecovery, RecoveryHook
from board_opt.tuner import TunedRuntimeConfig
from board_opt.vocab_prune import prune_vocab_gguf


@dataclass
class OptimizationReport:
    final_gguf: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_metrics: EvalMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_requant_targets(tensors: list[TensorInfo], hw: HardwareProfile) -> dict[str, int]:
    """Select non-native or oversized 2D/3D tensors to convert to the board's native sweet spot."""
    target_name = hw.preferred_formats[0] if hw.preferred_formats else "Q4_0"
    target_id = NAME_TO_TYPE.get(target_name, 2)
    targets: dict[str, int] = {}

    for t in tensors:
        # Never touch 1D norms, biases, or small vectors
        if len(t.dims) < 2 or t.dims[0] % 32 != 0:
            continue
        if "norm" in t.name or "bias" in t.name:
            continue
        # Non-native FP4/NVFP4 on boards without fp4 support
        if t.ttype in (39, 40) and not hw.fp4:
            targets[t.name] = target_id
            continue
        # Oversized F32 matrices (e.g. loop controllers)
        if t.ttype == 0:
            targets[t.name] = target_id
            continue
        # High-bit attention / output projections on bandwidth-bound boards
        is_attn_or_out = ("attn_" in t.name or t.name == "output.weight")
        if is_attn_or_out and t.ttype in (8, 13, 14):  # Q8_0, Q5_K, Q6_K
            targets[t.name] = target_id
    return targets


def plan_moe_expert_steps(
    info: GGUFInfo,
    vram_constrained: bool,
    prune_experts: int | None = None,
) -> dict[str, Any]:
    active_candidates: list[int] = []
    if info.expert_used_count >= 8:
        active_candidates = [6, 4]
    elif info.expert_used_count >= 6:
        active_candidates = [4]

    reap_target: int | None = None
    if info.expert_count >= 16:
        if prune_experts is not None and 0 < prune_experts < info.expert_count:
            reap_target = prune_experts
        elif vram_constrained:
            reap_target = max(info.expert_used_count * 2, (info.expert_count * 3) // 4)
    return {"active_candidates": active_candidates, "reap_target": reap_target}


def _select_reap_experts(scores_json: str, target_count: int) -> dict[int, list[int]]:
    with open(scores_json, encoding="utf-8") as f:
        data = json.load(f)
    keep: dict[int, list[int]] = {}
    layers = data.get("layers", data)
    for k, scores in layers.items():
        if not str(k).isdigit():
            continue
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep[int(k)] = sorted(ranked[:target_count])
    return keep


def optimize_model(
    info: GGUFInfo,
    hw: HardwareProfile,
    runtime: TunedRuntimeConfig,
    tc: ToolchainPaths | None,
    gate_cfg: GateConfig,
    vram_constrained: bool,
    work_dir: str,
    calib_file: str = "",
    domains: list[str] | None = None,
    langs: list[str] | None = None,
    eval_file: str | None = None,
    prune_experts: int | None = None,
    keep_intermediates: bool = False,
    recovery_hook: RecoveryHook | None = None,
    step_eval_fn: Callable[[str], EvalMetrics] | None = None,
) -> OptimizationReport:
    os.makedirs(work_dir, exist_ok=True)
    domains = domains or ["reasoning"]
    langs_list = langs or []

    current_gguf = info.path
    current_metrics = runtime.metrics or EvalMetrics(0.0, 0.0, 0.0, 0, 0, 0)
    steps: list[dict[str, Any]] = []
    intermediates: list[str] = []

    def _eval_gguf(path: str) -> EvalMetrics:
        if step_eval_fn is not None:
            return step_eval_fn(path)
        assert tc is not None
        est_mb = (os.path.getsize(path) // (1024 * 1024)) + 512
        return run_full_eval(
            tc=tc,
            gguf_path=path,
            calib_file=calib_file,
            domains=domains,
            langs=langs_list,
            ctk=runtime.ctk,
            ctv=runtime.ctv,
            ubatch=runtime.ubatch,
            context=runtime.context,
            top_k=runtime.top_k,
            vram_mb=est_mb,
            eval_file=eval_file,
            quick=False,
            env_vars=runtime.env_vars,
        )

    def _try_step(name: str, cand_path: str, force_accept_if_valid: bool = False) -> None:
        nonlocal current_gguf, current_metrics
        m = _eval_gguf(cand_path)
        dec = evaluate_candidate(current_metrics, m, gate_cfg, vram_constrained=vram_constrained)
        accepted = dec.accepted or (force_accept_if_valid and not m.degenerate and m.eval_passed >= current_metrics.eval_passed - gate_cfg.allow_eval_drop)
        steps.append({"step": name, "candidate": cand_path, "metrics": m.to_dict(), "decision": dec.to_dict(), "accepted": accepted})
        if accepted:
            if current_gguf != info.path and current_gguf not in intermediates:
                intermediates.append(current_gguf)
            current_gguf = cand_path
            current_metrics = m
        else:
            intermediates.append(cand_path)

    # Step 1: Metadata fix (top_k = 40) + hardware-native tensor re-quantization
    kv_fix: dict[str, tuple[int, Any]] = {}
    if info.top_k == 0 and info.vocab_size > 50000:
        kv_fix["general.sampling.top_k"] = (4, 40)
    req_targets = plan_requant_targets(info.tensors, hw)
    if kv_fix or req_targets:
        cand = os.path.join(work_dir, "step1_requant.gguf")
        edit_gguf(
            current_gguf,
            cand,
            kv_overrides=kv_fix,
            tensor_requant=req_targets,
            ggml_conv_bin=tc.ggml_conv if tc else None,
        )
        _try_step("metadata_and_requant", cand)

    # Step 2: Language/domain vocabulary pruning
    if langs_list:
        cand = os.path.join(work_dir, "step2_vocab_pruned.gguf")
        res = prune_vocab_gguf(current_gguf, cand, langs_list)
        if not res["skipped"]:
            _try_step("vocab_prune", cand)
        else:
            steps.append({"step": "vocab_prune", "skipped": True, "reason": res["reason"]})

    # Step 3: MoE active expert count sweep & REAP total-expert pruning
    cur_info = inspect_gguf(current_gguf)
    moe_plan = plan_moe_expert_steps(cur_info, vram_constrained=vram_constrained, prune_experts=prune_experts)
    for k_act in moe_plan["active_candidates"]:
        cand = os.path.join(work_dir, f"step3_active_e{k_act}.gguf")
        edit_gguf(current_gguf, cand, kv_overrides={f"{cur_info.arch}.expert_used_count": (4, k_act)})
        _try_step(f"moe_active_experts_{k_act}", cand)

    reap_target = moe_plan["reap_target"]
    if reap_target is not None and tc and tc.reap_trace and calib_file and os.path.isfile(calib_file):
        scores_p = os.path.join(work_dir, "reap_scores.json")
        subprocess.check_call([tc.reap_trace, current_gguf, calib_file, scores_p])
        keep_map = _select_reap_experts(scores_p, reap_target)
        cand = os.path.join(work_dir, f"step4_reap_e{reap_target}.gguf")
        edit_gguf(
            current_gguf,
            cand,
            kv_overrides={f"{cur_info.arch}.expert_count": (4, reap_target)},
            expert_keep=keep_map,
        )
        _try_step(f"moe_reap_prune_{reap_target}", cand)

    # Step 4: Phase 2 recovery hook (NoOp in Phase 1)
    hook = recovery_hook or NoOpRecovery()
    recovered = hook.run(current_gguf, _eval_gguf, hw)
    if recovered != current_gguf:
        _try_step("phase2_recovery", recovered)

    # Copy winning GGUF to stable final filename and clean intermediates
    final_gguf = os.path.join(work_dir, "model-optimized.gguf")
    if current_gguf == info.path:
        final_gguf = info.path
    elif os.path.abspath(current_gguf) != os.path.abspath(final_gguf):
        shutil.move(current_gguf, final_gguf)

    if not keep_intermediates:
        for p in intermediates:
            if p != info.path and p != final_gguf and os.path.exists(p):
                os.unlink(p)

    return OptimizationReport(final_gguf=final_gguf, steps=steps, final_metrics=current_metrics)
