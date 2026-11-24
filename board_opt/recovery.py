from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from board_opt.models import HardwareProfile


class RecoveryHook(ABC):
    """Phase 2 extension interface for post-quantization / post-pruning recovery (e.g. ES or QAT)."""

    @abstractmethod
    def run(
        self,
        gguf_path: str,
        eval_fn: Callable[[str], Any] | None,
        hw: HardwareProfile | None,
    ) -> str:
        """Return path to recovered GGUF (or gguf_path unchanged if no improvement)."""


class NoOpRecovery(RecoveryHook):
    def run(
        self,
        gguf_path: str,
        eval_fn: Callable[[str], Any] | None,
        hw: HardwareProfile | None,
    ) -> str:
        return gguf_path
