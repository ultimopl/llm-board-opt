from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HardwareProfile:
    cpu_arch: str
    cpu_march: str
    cpu_cores: int
    ram_total_mb: int
    usable_vram_mb: int
    backend: str  # "vulkan" | "rocm" | "cuda" | "cpu"
    gpu_name: str
    uma: bool
    fp16: bool
    bf16: bool
    fp4: bool
    int_dot: bool
    preferred_formats: list[str] = field(default_factory=list)
    warp_size: int = 32
    shared_memory_bytes: int = 65536

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareProfile:
        return cls(**d)


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ttype: int
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TensorInfo:
        return cls(**d)


@dataclass
class GGUFInfo:
    path: str
    arch: str
    file_size_bytes: int
    n_layer: int
    n_layer_kv: int
    n_head_kv: int
    head_k: int
    head_v: int
    vocab_size: int
    tokenizer_model: str
    expert_count: int
    expert_used_count: int
    top_k: int
    tensors: list[TensorInfo] = field(default_factory=list)
    kv_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GGUFInfo:
        d = dict(d)
        d["tensors"] = [TensorInfo.from_dict(t) if isinstance(t, dict) else t for t in d.get("tensors", [])]
        return cls(**d)


@dataclass
class BudgetCandidate:
    ctk: str
    ctv: str
    ubatch: int
    context: int
    estimated_vram_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BudgetCandidate:
        return cls(**d)


@dataclass
class BudgetPlan:
    usable_vram_mb: int
    weights_mb: int
    vram_constrained: bool
    recommended_max_context: int
    candidates: list[BudgetCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BudgetPlan:
        d = dict(d)
        d["candidates"] = [
            BudgetCandidate.from_dict(c) if isinstance(c, dict) else c
            for c in d.get("candidates", [])
        ]
        return cls(**d)


@dataclass
class EvalMetrics:
    pp_tps: float
    tg_tps: float
    ppl: float
    eval_passed: int
    eval_total: int
    vram_mb: int
    degenerate: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalMetrics:
        return cls(**d)


@dataclass
class GateConfig:
    allow_eval_drop: int = 0
    max_ppl_delta: float = 0.03
    min_tg_gain: float = 0.015
    min_vram_save_mib: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateConfig:
        return cls(**d)


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    tg_gain: float
    ppl_delta: float
    vram_saved_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateDecision:
        return cls(**d)
