from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass

from board_opt.models import HardwareProfile


@dataclass
class ToolchainPaths:
    llama_bench: str
    llama_perplexity: str
    llama_quantize: str
    llama_server: str
    ggml_conv: str | None = None
    reap_trace: str | None = None


def hardware_fingerprint(hw: HardwareProfile) -> str:
    key = f"{hw.cpu_arch}:{hw.cpu_march}:{hw.backend}:{hw.gpu_name}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def cmake_args_for_hardware(hw: HardwareProfile, src_dir: str, build_dir: str) -> list[str]:
    march = hw.cpu_march if hw.cpu_march and hw.cpu_march != "native" else "native"
    flags = f"-march={march} -mtune={march} -O3"
    args = [
        "cmake",
        "-S", src_dir,
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_FLAGS={flags}",
        f"-DCMAKE_CXX_FLAGS={flags}",
        "-DGGML_NATIVE=ON",
        "-DGGML_LTO=ON",
        "-DGGML_BACKEND_DL=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
    ]
    if shutil.which("ninja"):
        args.extend(["-G", "Ninja"])
    if hw.backend == "vulkan":
        args.append("-DGGML_VULKAN=ON")
    elif hw.backend == "rocm":
        args.append("-DGGML_HIP=ON")
    elif hw.backend == "cuda":
        args.append("-DGGML_CUDA=ON")
    return args


def _resolve_llama_src(llama_src: str | None, cache_dir: str) -> str:
    if llama_src:
        p = os.path.expanduser(llama_src)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"--llama-src directory not found: {p}")
        return p
    default_local = os.path.expanduser("~/src/llama-mainline")
    if os.path.isdir(default_local):
        return default_local
    cloned = os.path.join(cache_dir, "llama.cpp")
    if not os.path.isdir(cloned):
        os.makedirs(cache_dir, exist_ok=True)
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git", cloned,
        ])
    return cloned


def _build_native_helper(cpp_path: str, out_bin: str, src_dir: str, build_dir: str, link_llama: bool = False) -> str | None:
    if not os.path.exists(cpp_path) or not shutil.which("g++"):
        return None
    inc_ggml = os.path.join(src_dir, "ggml", "include")
    inc_llama = os.path.join(src_dir, "include")
    libs: list[str] = []
    for candidate in ("libllama.a", "libggml.a", "libggml-cpu.a", "libggml-vulkan.a", "libggml-hip.a", "libggml-cuda.a", "libggml-base.a"):
        if candidate == "libllama.a" and not link_llama:
            continue
        for root, _, files in os.walk(build_dir):
            if candidate in files:
                libs.append(os.path.join(root, candidate))
                break
    if not libs:
        return None
    cmd = ["g++", "-O3", "-flto", "-fopenmp", "-std=c++17", f"-I{inc_ggml}", f"-I{inc_llama}", cpp_path, "-o", out_bin]
    cmd.extend(["-Wl,--start-group", *libs, "-Wl,--end-group", "-lpthread", "-ldl", "-lm"])
    if shutil.which("vulkaninfo"):
        cmd.append("-lvulkan")
    try:
        subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
        return out_bin
    except subprocess.CalledProcessError:
        return None


def ensure_native_toolchain(
    hw: HardwareProfile,
    llama_src: str | None = None,
    cache_dir: str = "~/.cache/board-opt",
    rebuild: bool = False,
) -> ToolchainPaths:
    cache_root = os.path.expanduser(cache_dir)
    fp = hardware_fingerprint(hw)
    build_dir = os.path.join(cache_root, f"llama-native-{fp}")
    bin_dir = os.path.join(build_dir, "bin")

    expected = {
        "llama_bench": os.path.join(bin_dir, "llama-bench"),
        "llama_perplexity": os.path.join(bin_dir, "llama-perplexity"),
        "llama_quantize": os.path.join(bin_dir, "llama-quantize"),
        "llama_server": os.path.join(bin_dir, "llama-server"),
    }

    if rebuild or not all(os.path.isfile(p) for p in expected.values()):
        src_dir = _resolve_llama_src(llama_src, cache_root)
        args = cmake_args_for_hardware(hw, src_dir, build_dir)
        subprocess.check_call(args)
        subprocess.check_call([
            "cmake", "--build", build_dir,
            "--target", "llama-bench", "llama-perplexity", "llama-quantize", "llama-server",
            "-j", str(max(1, hw.cpu_cores)),
        ])
    else:
        src_dir = _resolve_llama_src(llama_src, cache_root)

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    native_dir = os.path.join(pkg_root, "native")
    ggml_conv = os.path.join(bin_dir, "ggml_conv")
    reap_trace = os.path.join(bin_dir, "reap_trace")

    if os.path.isdir(src_dir):
        if rebuild or not os.path.isfile(ggml_conv):
            ggml_conv = _build_native_helper(os.path.join(native_dir, "ggml_conv.cpp"), ggml_conv, src_dir, build_dir, link_llama=False)
        if rebuild or not os.path.isfile(reap_trace):
            reap_trace = _build_native_helper(os.path.join(native_dir, "reap_trace.cpp"), reap_trace, src_dir, build_dir, link_llama=True)

    return ToolchainPaths(
        llama_bench=expected["llama_bench"],
        llama_perplexity=expected["llama_perplexity"],
        llama_quantize=expected["llama_quantize"],
        llama_server=expected["llama_server"],
        ggml_conv=ggml_conv if ggml_conv and os.path.isfile(ggml_conv) else None,
        reap_trace=reap_trace if reap_trace and os.path.isfile(reap_trace) else None,
    )
