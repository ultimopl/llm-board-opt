from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from board_opt.models import EvalMetrics, GateConfig, GateDecision


def evaluate_candidate(
    baseline: EvalMetrics,
    cand: EvalMetrics,
    cfg: GateConfig,
    vram_constrained: bool,
) -> GateDecision:
    tg_gain = (cand.tg_tps - baseline.tg_tps) / baseline.tg_tps if baseline.tg_tps > 0 else 0.0
    ppl_delta = (cand.ppl - baseline.ppl) / baseline.ppl if baseline.ppl > 0 else 0.0
    vram_saved = baseline.vram_mb - cand.vram_mb

    if cand.degenerate:
        return GateDecision(False, "degenerate output detected", tg_gain, ppl_delta, vram_saved)

    if cand.eval_total > 0 and baseline.eval_total > 0:
        min_allowed = max(0, baseline.eval_passed - cfg.allow_eval_drop)
        if cand.eval_passed < min_allowed:
            return GateDecision(
                False,
                f"eval regression: {cand.eval_passed}/{cand.eval_total} < {min_allowed}/{baseline.eval_total}",
                tg_gain,
                ppl_delta,
                vram_saved,
            )

    if baseline.ppl > 0 and cand.ppl > 0 and ppl_delta > cfg.max_ppl_delta:
        return GateDecision(
            False,
            f"perplexity delta {ppl_delta:+.2%} exceeds max {cfg.max_ppl_delta:+.2%}",
            tg_gain,
            ppl_delta,
            vram_saved,
        )

    meets_speed = tg_gain >= cfg.min_tg_gain
    meets_vram = vram_constrained and (vram_saved >= cfg.min_vram_save_mib)

    if meets_speed or meets_vram:
        why = []
        if meets_speed:
            why.append(f"tg {tg_gain:+.1%}")
        if meets_vram:
            why.append(f"vram -{vram_saved} MiB")
        return GateDecision(True, "accepted (" + ", ".join(why) + ")", tg_gain, ppl_delta, vram_saved)

    return GateDecision(
        False,
        f"insufficient gain (tg {tg_gain:+.1%} < {cfg.min_tg_gain:+.1%}, vram_saved={vram_saved} MiB)",
        tg_gain,
        ppl_delta,
        vram_saved,
    )


class StageStore:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

    def path_for(self, stage: str) -> str:
        return os.path.join(self.work_dir, f"{stage}.json")

    def save(self, stage: str, data: dict[str, Any]) -> str:
        dst = self.path_for(stage)
        fd, tmp = tempfile.mkstemp(dir=self.work_dir, prefix=f".{stage}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, dst)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return dst

    def load(self, stage: str) -> dict[str, Any] | None:
        p = self.path_for(stage)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)
