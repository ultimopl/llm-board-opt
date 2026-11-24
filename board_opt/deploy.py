from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import time
import urllib.request
from typing import Any

from board_opt.builder import ToolchainPaths
from board_opt.tuner import TunedRuntimeConfig

GPU_PROCS = ("llama-server", "llama-bench", "llama-perplexity", "llama-cli")


def check_active_gpu_procs(lines: list[str]) -> None:
    own_pids = {str(os.getpid()), str(os.getppid())}
    hits: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        pid = s.split(None, 1)[0]
        if pid in own_pids:
            continue
        if any(p in s for p in GPU_PROCS):
            hits.append(s)
    if hits:
        raise RuntimeError(
            "Active GPU process detected (strictly one GPU job at a time): "
            + "; ".join(hits)
            + ". Stop it before running board-opt."
        )


def extract_smoke_reply(body: dict[str, Any]) -> str:
    msg = body["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return str(text).strip()[:80]


def ensure_single_gpu_job(managed_unit: str | None = None) -> None:
    if managed_unit:
        subprocess.run(
            ["systemctl", "--user", "stop", f"{managed_unit}.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    res = subprocess.run(
        ["pgrep", "-af", "llama-(server|bench|perplexity|cli)"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0 and res.stdout.strip():
        check_active_gpu_procs(res.stdout.splitlines())


def generate_serve_script(
    gguf_path: str,
    runtime: TunedRuntimeConfig,
    tc: ToolchainPaths,
    host: str,
    port: int,
    out_path: str,
) -> str:
    env_lines = [f"export {k}={shlex.quote(str(v))}" for k, v in sorted(runtime.env_vars.items())]
    env_block = "\n".join(env_lines) + ("\n" if env_lines else "")
    fa_val = "on" if runtime.flash_attn else "off"

    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{env_block}"
        f"exec {shlex.quote(tc.llama_server)} \\\n"
        f"  -m {shlex.quote(os.path.abspath(gguf_path))} \\\n"
        f"  --host {shlex.quote(host)} --port {int(port)} \\\n"
        f"  -fa {fa_val} -ngl 99 \\\n"
        f"  -c \"${{1:-{int(runtime.context)}}}\" \\\n"
        f"  -b 512 -ub {int(runtime.ubatch)} \\\n"
        f"  -ctk {shlex.quote(runtime.ctk)} -ctv {shlex.quote(runtime.ctv)} \\\n"
        f"  -np 1 --cache-ram {int(runtime.cache_ram)} \\\n"
        f"  --top-k {int(runtime.top_k)} --jinja\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script)
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out_path


def launch_and_verify_server(serve_script: str, unit_name: str, port: int) -> dict[str, Any]:
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit_name}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["systemctl", "--user", "reset-failed", f"{unit_name}.service"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call([
        "systemd-run", "--user", "--collect", f"--unit={unit_name}", os.path.abspath(serve_script),
    ])

    healthy = False
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            time.sleep(0.5)

    if not healthy:
        raise RuntimeError(f"Server unit '{unit_name}' failed to become healthy on port {port}")

    payload = json.dumps({
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 16,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    reply = extract_smoke_reply(body)
    return {"unit": unit_name, "port": port, "healthy": True, "smoke_reply": reply}
