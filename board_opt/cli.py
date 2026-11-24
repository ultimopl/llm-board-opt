from __future__ import annotations

import argparse
import json
import os
import re
import sys

from board_opt.budget import solve_budget
from board_opt.builder import ensure_native_toolchain
from board_opt.deploy import (
    ensure_single_gpu_job,
    generate_serve_script,
    launch_and_verify_server,
)
from board_opt.eval_runner import build_calibration_file, run_full_eval
from board_opt.gate import StageStore
from board_opt.gguf_io import inspect_gguf
from board_opt.inspector import (
    format_inspection_report,
    summarize_layer_quants,
    summarize_moe_experts,
    summarize_vocab_scripts,
)
from board_opt.log import log_info, set_verbosity
from board_opt.models import EvalMetrics, GateConfig, HardwareProfile
from board_opt.optimizer import optimize_model
from board_opt.probe import probe_hardware
from board_opt.quant_bench import QuantBenchEntry, format_quant_table, run_quant_bench
from board_opt.tuner import TunedRuntimeConfig, tune_runtime


def _slugify(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-").lower() or "model"


def validate_budget_or_abort(plan, requested_ctx: int) -> None:
    if not plan.candidates and plan.recommended_max_context <= 0:
        raise RuntimeError(
            f"Model weights ({plan.weights_mb} MiB) exceeds usable VRAM ({plan.usable_vram_mb} MiB) "
            f"for requested context {requested_ctx}; cannot launch GPU jobs without OOM. "
            "Use a smaller quant or lower --context."
        )


def _cmd_probe(args: argparse.Namespace) -> int:
    hw = probe_hardware(vram_reserve_mb=args.vram_reserve_mb, backend_override=args.backend)
    print(json.dumps(hw.to_dict(), indent=2))
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    hw = probe_hardware(vram_reserve_mb=args.vram_reserve_mb, backend_override=args.backend)
    info = inspect_gguf(args.model)
    plan = solve_budget(info, hw, requested_ctx=args.context, target_vram_mb=args.target_vram_mb)
    out = {
        "hardware": hw.to_dict(),
        "model": {
            "arch": info.arch,
            "file_size_mb": info.file_size_bytes // (1024 * 1024),
            "n_layer": info.n_layer,
            "n_layer_kv": info.n_layer_kv,
            "vocab_size": info.vocab_size,
            "expert_count": info.expert_count,
            "expert_used_count": info.expert_used_count,
            "top_k": info.top_k,
        },
        "budget": plan.to_dict(),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    set_verbosity(0 if args.quiet else (2 if args.verbose else 1))
    model_path = os.path.abspath(os.path.expanduser(args.model))
    slug = _slugify(model_path)
    work_dir = os.path.abspath(os.path.join(os.path.expanduser(args.work_dir), slug))
    store = StageStore(work_dir)
    unit_name = args.unit_name or f"board-opt-{slug[:24]}"

    ensure_single_gpu_job(managed_unit=unit_name)

    # Stage 0: Hardware probe
    log_info(f"[Stage 0] Probing hardware (work_dir={work_dir})...")
    hw_data = store.load("hardware")
    if hw_data and not args.no_resume:
        hw = HardwareProfile.from_dict(hw_data)
    else:
        hw = probe_hardware(vram_reserve_mb=args.vram_reserve_mb, backend_override=args.backend)
        store.save("hardware", hw.to_dict())
    log_info(f"[Stage 0] {hw.gpu_name} | backend={hw.backend} | usable_vram={hw.usable_vram_mb} MiB | march={hw.cpu_march}")

    # Stage 1: GGUF inspect & VRAM/KV budget
    info = inspect_gguf(model_path)
    plan = solve_budget(info, hw, requested_ctx=args.context, target_vram_mb=args.target_vram_mb)
    store.save("plan", plan.to_dict())
    log_info(f"[Stage 1] {info.arch} ({plan.weights_mb} MiB, {info.n_layer_kv} full KV layers, {len(plan.candidates)} budget-valid candidates)")
    validate_budget_or_abort(plan, args.context)

    # Stage 2: Native board compilation
    log_info("[Stage 2] Ensuring board-native llama.cpp toolchain...")
    tc = ensure_native_toolchain(hw, llama_src=args.llama_src, rebuild=args.rebuild)

    # Stage 2b: Empirical per-quant board benchmark (speed, GB/s, conversion speed, RMSE/CosSim)
    quant_table: dict[str, QuantBenchEntry] | None = None
    qb_data = store.load("quant_bench")
    if qb_data and not args.no_resume:
        quant_table = {k: QuantBenchEntry.from_dict(v) for k, v in qb_data.items()}
    elif tc.quant_bench:
        log_info("[Stage 2b] Benchmarking all GGML quant formats on this board (speed, GB/s, conv, RMSE/CosSim)...")
        quant_table = run_quant_bench(tc.quant_bench)
        store.save("quant_bench", {k: v.to_dict() for k, v in quant_table.items()})
        log_info("\n" + format_quant_table(quant_table))

    langs = [x.strip() for x in args.langs.split(",") if x.strip()] if args.langs else []
    domains = [x.strip() for x in args.domains.split(",") if x.strip()] if args.domains else ["reasoning"]
    calib_file = build_calibration_file(
        langs or ["en"], domains, os.path.join(work_dir, "calibration.txt"), custom_ppl_file=args.ppl_file
    )

    gate_cfg = GateConfig(
        allow_eval_drop=args.allow_eval_drop,
        max_ppl_delta=args.max_ppl_delta,
        min_tg_gain=args.min_tg_gain,
        min_vram_save_mib=args.min_vram_save_mib,
    )

    # Stage 3: Baseline lock
    base_data = store.load("baseline")
    if base_data and not args.no_resume:
        baseline = EvalMetrics.from_dict(base_data)
        log_info(f"[Stage 3] Resumed baseline: pp={baseline.pp_tps:.1f} tg={baseline.tg_tps:.1f} ppl={baseline.ppl:.3f} eval={baseline.eval_passed}/{baseline.eval_total}")
    else:
        log_info("[Stage 3] Running baseline bench + perplexity + domain evals...")
        first_c = plan.candidates[0] if plan.candidates else None
        baseline = run_full_eval(
            tc=tc,
            gguf_path=model_path,
            calib_file=calib_file,
            domains=domains,
            langs=langs,
            ctk=first_c.ctk if first_c else "q4_0",
            ctv=first_c.ctv if first_c else "q4_0",
            ubatch=first_c.ubatch if first_c else 128,
            context=first_c.context if first_c else plan.recommended_max_context,
            top_k=info.top_k if info.top_k is not None else 40,
            vram_mb=first_c.estimated_vram_mb if first_c else plan.weights_mb,
            eval_file=args.eval_file,
            quick=False,
        )
        store.save("baseline", baseline.to_dict())
        log_info(f"[Stage 3] Baseline: pp={baseline.pp_tps:.1f} tg={baseline.tg_tps:.1f} ppl={baseline.ppl:.3f} eval={baseline.eval_passed}/{baseline.eval_total}")

    # Stage 4: Runtime & KV permutation sweep
    tune_data = store.load("tune")
    if tune_data and not args.no_resume:
        runtime = TunedRuntimeConfig.from_dict(tune_data)
        log_info(f"[Stage 4] Resumed tuned runtime: {runtime.ctk}/{runtime.ctv} ub={runtime.ubatch}")
    else:
        runtime = tune_runtime(
            info=info,
            hw=hw,
            plan=plan,
            tc=tc,
            baseline=baseline,
            gate_cfg=gate_cfg,
            domains=domains,
            langs=langs,
            calib_file=calib_file,
            eval_file=args.eval_file,
        )
        store.save("tune", runtime.to_dict())
        log_info(f"[Stage 4] Selected runtime: {runtime.ctk}/{runtime.ctv} ub={runtime.ubatch} top_k={runtime.top_k}")

    # Stage 5: Automatic GGUF optimization
    log_info("[Stage 5] Running greedy GGUF optimization...")
    cut_scripts = [x.strip() for x in args.cut_scripts.split(",") if x.strip()] if args.cut_scripts else None
    opt_report = optimize_model(
        info=info,
        hw=hw,
        runtime=runtime,
        tc=tc,
        gate_cfg=gate_cfg,
        vram_constrained=plan.vram_constrained,
        work_dir=work_dir,
        calib_file=calib_file,
        domains=domains,
        langs=langs,
        cut_scripts=cut_scripts,
        eval_file=args.eval_file,
        prune_experts=args.prune_experts,
        keep_intermediates=args.keep_intermediates,
        quant_table=quant_table,
    )
    store.save("optimize", opt_report.to_dict())

    # Stage 6: Packaging & optional server launch
    log_info("[Stage 6] Packaging serve.sh" + (" and launching systemd server..." if args.serve else "..."))
    serve_script = generate_serve_script(
        opt_report.final_gguf, runtime, tc, args.host, args.port, os.path.join(work_dir, "serve.sh")
    )
    server_info = None
    if args.serve:
        server_info = launch_and_verify_server(serve_script, unit_name, args.port)

    final = {
        "model_input": model_path,
        "model_output": opt_report.final_gguf,
        "serve_script": serve_script,
        "baseline": baseline.to_dict(),
        "final_metrics": opt_report.final_metrics.to_dict() if opt_report.final_metrics else None,
        "runtime": runtime.to_dict(),
        "server": server_info,
    }
    store.save("final", final)
    print(json.dumps(final, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    hw = probe_hardware(vram_reserve_mb=args.vram_reserve_mb, backend_override=args.backend)
    info = inspect_gguf(args.model)
    plan = solve_budget(info, hw, requested_ctx=args.context, target_vram_mb=args.target_vram_mb)

    quant_table = None
    if args.quant_bench_json and os.path.isfile(args.quant_bench_json):
        with open(args.quant_bench_json, encoding="utf-8") as f:
            raw = json.load(f)
        quant_table = {k: QuantBenchEntry.from_dict(v) for k, v in raw.items()}

    layer_rows = summarize_layer_quants(info, hw, quant_table=quant_table, vram_constrained=plan.vram_constrained)
    moe = summarize_moe_experts(info, vram_constrained=plan.vram_constrained)
    vocab = summarize_vocab_scripts(args.model)

    if args.json:
        print(json.dumps({"layers": layer_rows, "moe": moe, "vocab": vocab}, indent=2))
    else:
        print(f"Model: {os.path.basename(args.model)} ({info.arch}, {plan.weights_mb} MiB) | Board: {hw.gpu_name} ({hw.usable_vram_mb} MiB usable)")
        print(format_inspection_report(layer_rows, moe, vocab))
    return 0


def _cmd_bench_quants(args: argparse.Namespace) -> int:
    ensure_single_gpu_job(managed_unit=args.unit_name)
    hw = probe_hardware(vram_reserve_mb=args.vram_reserve_mb, backend_override=args.backend)
    tc = ensure_native_toolchain(hw, llama_src=args.llama_src, rebuild=args.rebuild)
    if not tc.quant_bench:
        raise RuntimeError("quant_bench native helper failed to build (check g++ and llama.cpp build dir)")
    table = run_quant_bench(tc.quant_bench, m=args.m, k=args.k)
    if args.json:
        print(json.dumps({k: v.to_dict() for k, v in table.items()}, indent=2))
    else:
        print(f"Board: {hw.gpu_name} ({hw.backend}, march={hw.cpu_march}) | Matrix: {args.m}x{args.k}")
        print(format_quant_table(table))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    serve_script = os.path.join(work_dir, "serve.sh")
    if not os.path.isfile(serve_script):
        raise FileNotFoundError(f"serve.sh not found in {work_dir}; run `board-opt run` first.")
    unit_name = args.unit_name or f"board-opt-{os.path.basename(work_dir)[:24]}"
    ensure_single_gpu_job(managed_unit=unit_name)
    res = launch_and_verify_server(serve_script, unit_name, args.port)
    print(json.dumps(res, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="board-opt", description="Board-specific LLM optimization framework")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_probe = sub.add_parser("probe", help="Probe local board hardware")
    sp_probe.add_argument("--vram-reserve-mb", type=int, default=None)
    sp_probe.add_argument("--backend", choices=["vulkan", "rocm", "cuda", "cpu"], default=None)
    sp_probe.set_defaults(func=_cmd_probe)

    sp_budget = sub.add_parser("budget", help="Inspect GGUF and solve VRAM/KV budget")
    sp_budget.add_argument("--model", required=True)
    sp_budget.add_argument("--context", type=int, default=32768)
    sp_budget.add_argument("--target-vram-mb", type=int, default=None)
    sp_budget.add_argument("--vram-reserve-mb", type=int, default=None)
    sp_budget.add_argument("--backend", choices=["vulkan", "rocm", "cuda", "cpu"], default=None)
    sp_budget.set_defaults(func=_cmd_budget)

    sp_run = sub.add_parser("run", help="Run full optimization pipeline")
    sp_run.add_argument("--model", required=True)
    sp_run.add_argument("--langs", default="", help="Comma-separated languages to keep for vocab pruning (e.g. en,pt)")
    sp_run.add_argument("--cut-scripts", default="", help="Explicit scripts to cut from vocab (e.g. cjk,cyrillic,arabic,other_unicode)")
    sp_run.add_argument("--domains", default="reasoning,python,c", help="Comma-separated eval domains")
    sp_run.add_argument("--context", type=int, default=32768)
    sp_run.add_argument("--target-vram-mb", type=int, default=None)
    sp_run.add_argument("--vram-reserve-mb", type=int, default=None)
    sp_run.add_argument("--backend", choices=["vulkan", "rocm", "cuda", "cpu"], default=None)
    sp_run.add_argument("--llama-src", default=None)
    sp_run.add_argument("--rebuild", action="store_true")
    sp_run.add_argument("--ppl-file", default=None)
    sp_run.add_argument("--eval-file", default=None)
    sp_run.add_argument("--prune-experts", type=int, default=None)
    sp_run.add_argument("--allow-eval-drop", type=int, default=0)
    sp_run.add_argument("--max-ppl-delta", type=float, default=0.03)
    sp_run.add_argument("--min-tg-gain", type=float, default=0.015)
    sp_run.add_argument("--min-vram-save-mib", type=int, default=100)
    sp_run.add_argument("--work-dir", default="./work")
    sp_run.add_argument("--keep-intermediates", action="store_true")
    sp_run.add_argument("--no-resume", action="store_true")
    sp_run.add_argument("--serve", action="store_true")
    sp_run.add_argument("--host", default="0.0.0.0")
    sp_run.add_argument("--port", type=int, default=8080)
    sp_run.add_argument("--unit-name", default=None)
    sp_run.add_argument("-v", "--verbose", action="store_true", help="Print per-candidate and per-task progress to stderr")
    sp_run.add_argument("-q", "--quiet", action="store_true", help="Suppress progress logs (JSON stdout only)")
    sp_run.set_defaults(func=_cmd_run)

    sp_insp = sub.add_parser("inspect", help="Show layer quants, prunable MoE experts, vocab scripts, and expected gains/losses")
    sp_insp.add_argument("--model", required=True)
    sp_insp.add_argument("--context", type=int, default=32768)
    sp_insp.add_argument("--target-vram-mb", type=int, default=None)
    sp_insp.add_argument("--vram-reserve-mb", type=int, default=None)
    sp_insp.add_argument("--backend", choices=["vulkan", "rocm", "cuda", "cpu"], default=None)
    sp_insp.add_argument("--quant-bench-json", default=None, help="Optional cached quant_bench.json for exact board speed/loss estimates")
    sp_insp.add_argument("--json", action="store_true")
    sp_insp.set_defaults(func=_cmd_inspect)

    sp_bq = sub.add_parser("bench-quants", help="Benchmark all GGML quant formats on this board (speed, GB/s, conv, loss)")
    sp_bq.add_argument("--m", type=int, default=4096)
    sp_bq.add_argument("--k", type=int, default=4096)
    sp_bq.add_argument("--llama-src", default=None)
    sp_bq.add_argument("--rebuild", action="store_true")
    sp_bq.add_argument("--vram-reserve-mb", type=int, default=None)
    sp_bq.add_argument("--backend", choices=["vulkan", "rocm", "cuda", "cpu"], default=None)
    sp_bq.add_argument("--unit-name", default=None)
    sp_bq.add_argument("--json", action="store_true")
    sp_bq.set_defaults(func=_cmd_bench_quants)

    sp_serve = sub.add_parser("serve", help="Launch previously generated serve.sh via systemd-run")
    sp_serve.add_argument("--work-dir", required=True)
    sp_serve.add_argument("--port", type=int, default=8080)
    sp_serve.add_argument("--unit-name", default=None)
    sp_serve.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
