from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QuantBenchEntry:
    name: str
    type_id: int
    supported: bool
    bpw: float
    tg_ms: float
    tg_gws: float
    tg_gbps: float
    pp_gws: float
    conv_ms_per_m: float
    rmse: float
    cos_sim: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuantBenchEntry:
        return cls(**d)


def parse_quant_bench_json(raw_json: str) -> dict[str, QuantBenchEntry]:
    data = json.loads(raw_json)
    out: dict[str, QuantBenchEntry] = {}
    for row in data.get("results", []):
        e = QuantBenchEntry.from_dict(row)
        out[e.name.upper()] = e
    return out


def run_quant_bench(quant_bench_bin: str, m: int = 4096, k: int = 4096) -> dict[str, QuantBenchEntry]:
    out = subprocess.check_output([quant_bench_bin, str(m), str(k)], text=True, stderr=subprocess.DEVNULL)
    return parse_quant_bench_json(out)


def pick_best_target_format(
    table: dict[str, QuantBenchEntry],
    src_type: int,
    min_cos_sim: float = 0.995,
    vram_constrained: bool = False,
    min_speedup_ratio: float = 1.05,
) -> QuantBenchEntry | None:
    """Empirically pick the fastest supported quant format on this board meeting quality loss bounds."""
    src_entry = next((e for e in table.values() if e.type_id == src_type), None)
    src_gws = src_entry.tg_gws if (src_entry and src_entry.supported) else 0.0
    src_bpw = src_entry.bpw if src_entry else 32.0

    eligible = [
        e for e in table.values()
        if e.supported and e.tg_gws > 0 and e.cos_sim >= min_cos_sim
    ]
    if not eligible:
        return None

    # Sort by highest decode speed (GW/s), breaking ties by lower RMSE
    eligible.sort(key=lambda e: (-e.tg_gws, e.rmse))
    best = eligible[0]
    if best.type_id == src_type:
        return None
    if src_gws > 0 and best.tg_gws < src_gws * min_speedup_ratio:
        if not (vram_constrained and best.bpw < src_bpw * 0.85):
            return None
    return best


def format_quant_table(table: dict[str, QuantBenchEntry]) -> str:
    rows = sorted(table.values(), key=lambda e: (-e.tg_gws if e.supported else 1.0, e.bpw))
    lines = [
        f"{'Quant':<10} | {'Sup':^3} | {'BPW':>5} | {'tg ms':>7} | {'tg GW/s':>8} | {'tg GB/s':>8} | {'pp GW/s':>8} | {'conv ms/M':>9} | {'RMSE':>8} | {'CosSim':>8}",
        "-" * 102,
    ]
    for e in rows:
        lines.append(
            f"{e.name:<10} | {'Y' if e.supported else 'N':^3} | {e.bpw:5.2f} | {e.tg_ms:7.3f} | "
            f"{e.tg_gws:8.1f} | {e.tg_gbps:8.1f} | {e.pp_gws:8.1f} | {e.conv_ms_per_m:9.2f} | "
            f"{e.rmse:8.5f} | {e.cos_sim:8.6f}"
        )
    return "\n".join(lines)
