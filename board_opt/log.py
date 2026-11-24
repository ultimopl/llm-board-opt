from __future__ import annotations

import sys
import time

_VERBOSITY = 1  # 0=quiet, 1=stage progress (default), 2=verbose (per-candidate/per-task)


def set_verbosity(level: int) -> None:
    global _VERBOSITY
    _VERBOSITY = level


def get_verbosity() -> int:
    return _VERBOSITY


def log_info(msg: str) -> None:
    if _VERBOSITY >= 1:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def log_verbose(msg: str) -> None:
    if _VERBOSITY >= 2:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}]   {msg}", file=sys.stderr, flush=True)
