#!/usr/bin/env python3
"""
Objective LiveCodeBench V6 evaluator for comparing baseline vs optimized GGUF models
against an OpenAI-compatible llama-server endpoint (local, BC-250, or Colab).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request


def extract_python(text: str) -> str:
    m = re.findall(r"```python\b\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m2 = re.findall(r"```[a-zA-Z0-9_+-]*\s*(.*?)```", text, flags=re.DOTALL)
    if m2:
        return m2[-1].strip()
    return text.strip()


def grade_stdin_task(code: str, tests: list[dict], timeout_s: int = 5) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        sol_p = os.path.join(td, "sol.py")
        with open(sol_p, "w", encoding="utf-8") as f:
            f.write(code)
        for idx, tc in enumerate(tests):
            inp = str(tc.get("input", ""))
            exp = str(tc.get("output", "")).strip()
            try:
                r = subprocess.run(
                    ["python3", sol_p],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return False, f"tc {idx}: timeout"
            if r.returncode != 0:
                return False, f"tc {idx}: runtime error: {r.stderr.strip()[:120]}"
            got = r.stdout.strip()
            if got != exp:
                return False, f"tc {idx}: got {got[:60]!r}, expected {exp[:60]!r}"
    return True, "ok"


def build_prompt(task: dict) -> str:
    base = task["question_content"].strip()
    if task.get("starter_code"):
        return (
            f"{base}\n\nComplete the following Python starter code and return ONLY a ```python``` block:\n"
            f"```python\n{task['starter_code']}\n```"
        )
    return (
        f"{base}\n\nWrite a complete Python 3 program that reads from standard input (stdin) "
        f"and writes the exact answer to standard output (stdout). Return ONLY a ```python``` block."
    )


def call_server(url: str, prompt: str, max_tokens: int) -> tuple[str, float, float]:
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.monotonic() - t0
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    timings = body.get("timings", {})
    pp = float(timings.get("prompt_per_second", 0.0))
    tg = float(timings.get("predicted_per_second", 0.0))
    if tg == 0.0:
        n_tok = body.get("usage", {}).get("completion_tokens", 0)
        tg = n_tok / max(0.01, dt)
    return content, pp, tg


def main() -> int:
    ap = argparse.ArgumentParser(description="Run LiveCodeBench V6 against a llama-server endpoint")
    ap.add_argument("--tasks", default=os.path.join(os.path.dirname(__file__), "sensenova_lcb_v6_25.json"))
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = json.load(open(args.tasks, encoding="utf-8"))
    # Filter to stdin/stdout problems for uniform grading
    stdin_tasks = [t for t in tasks if not t.get("starter_code")][: args.limit]
    if not stdin_tasks:
        stdin_tasks = tasks[: args.limit]

    results = []
    passed = 0
    pps, tgs = [], []
    t_start = time.monotonic()

    for idx, t in enumerate(stdin_tasks, 1):
        prompt = build_prompt(t)
        content, pp, tg = call_server(args.url, prompt, args.max_tokens)
        code = extract_python(content)
        ok, note = grade_stdin_task(code, t["public_test_cases"])
        if ok:
            passed += 1
        if pp > 0:
            pps.append(pp)
        if tg > 0:
            tgs.append(tg)
        print(f"[{idx:02d}/{len(stdin_tasks):02d}] {t['id']:<18} ({t['difficulty']:<6}) -> {'PASS' if ok else 'FAIL'} | tg={tg:5.1f} t/s | {note}", flush=True)
        results.append({"id": t["id"], "difficulty": t["difficulty"], "passed": ok, "note": note, "pp": pp, "tg": tg})

    wall = time.monotonic() - t_start
    summary = {
        "passed": passed,
        "total": len(stdin_tasks),
        "pass_rate": round(100.0 * passed / max(1, len(stdin_tasks)), 1),
        "avg_pp": round(sum(pps) / max(1, len(pps)), 2),
        "avg_tg": round(sum(tgs) / max(1, len(tgs)), 2),
        "wall_s": round(wall, 1),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY: {passed}/{len(stdin_tasks)} ({summary['pass_rate']}%) | avg_pp={summary['avg_pp']} t/s | avg_tg={summary['avg_tg']} t/s | wall={summary['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
