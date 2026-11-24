from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Any

from board_opt.models import HardwareProfile


def compute_usable_vram_mb(total_mb: int, reserve_mb: int | None = None) -> int:
    if reserve_mb is None:
        reserve_mb = max(1536, int(total_mb * 0.10))
    return max(0, total_mb - reserve_mb)


def parse_vulkaninfo(text: str) -> dict[str, Any]:
    gpu_name = "unknown"
    uma = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("deviceName"):
            m = re.search(r"=\s*(.+)$", s)
            if m:
                candidate = m.group(1).strip()
                if "llvmpipe" not in candidate.lower() or gpu_name == "unknown":
                    gpu_name = candidate
        elif s.startswith("deviceType"):
            if "INTEGRATED_GPU" in s:
                uma = True
    if "BC-250" in gpu_name or "GFX1013" in gpu_name.upper():
        uma = True
    return {"gpu_name": gpu_name, "uma": uma}


def select_native_formats(profile: HardwareProfile) -> list[str]:
    if profile.fp4:
        return ["NVFP4", "Q4_0", "Q8_0", "F16"]
    if profile.int_dot:
        return ["Q4_0", "Q4_K", "Q8_0", "F16"]
    return ["Q4_0", "Q8_0", "F16"]


def _read_ram_total_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except OSError:
        pass
    return 8192


def _detect_cpu_march() -> str:
    try:
        out = subprocess.check_output(["gcc", "-march=native", "-Q", "--help=target"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "-march=" in line and "[enabled]" not in line:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == "-march=":
                    return parts[1]
    except Exception:
        pass
    return "native"


def _run_cmd(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None


def probe_hardware(
    vram_reserve_mb: int | None = None,
    backend_override: str | None = None,
) -> HardwareProfile:
    cpu_arch = platform.machine() or "x86_64"
    cpu_march = _detect_cpu_march()
    cpu_cores = os.cpu_count() or 1
    ram_total_mb = _read_ram_total_mb()

    backend = backend_override
    gpu_name = "CPU"
    uma = True
    vram_total_mb = ram_total_mb
    fp16 = True
    bf16 = False
    fp4 = False
    int_dot = False

    vk_out = _run_cmd(["vulkaninfo", "--summary"])
    nv_out = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    rocm_out = _run_cmd(["rocminfo"])

    if backend is None:
        if nv_out and nv_out.strip():
            backend = "cuda"
        elif vk_out and "deviceName" in vk_out:
            backend = "vulkan"
        elif rocm_out and "gfx" in rocm_out.lower():
            backend = "rocm"
        else:
            backend = "cpu"

    if backend == "cuda" and nv_out and nv_out.strip():
        first = nv_out.strip().splitlines()[0].split(",")
        gpu_name = first[0].strip()
        if len(first) > 1 and first[1].strip().isdigit():
            vram_total_mb = int(first[1].strip())
        uma = False
        bf16 = True
        int_dot = True
    elif backend in ("vulkan", "rocm") and vk_out:
        vinfo = parse_vulkaninfo(vk_out)
        gpu_name = vinfo["gpu_name"]
        uma = vinfo["uma"]
        if "GFX1013" in gpu_name.upper() or "BC-250" in gpu_name.upper():
            fp16 = True
            bf16 = False
            fp4 = False
            int_dot = False

    usable_vram_mb = compute_usable_vram_mb(vram_total_mb, vram_reserve_mb)
    hw = HardwareProfile(
        cpu_arch=cpu_arch,
        cpu_march=cpu_march,
        cpu_cores=cpu_cores,
        ram_total_mb=ram_total_mb,
        usable_vram_mb=usable_vram_mb,
        backend=backend,
        gpu_name=gpu_name,
        uma=uma,
        fp16=fp16,
        bf16=bf16,
        fp4=fp4,
        int_dot=int_dot,
    )
    hw.preferred_formats = select_native_formats(hw)
    return hw
