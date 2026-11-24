from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import tempfile
import time
import urllib.request
from typing import Any

from board_opt.builder import ToolchainPaths
from board_opt.models import EvalMetrics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BUILTIN_TASKS: list[dict[str, Any]] = [
    # --- REASONING ---
    {
        "id": "reason_mod",
        "domain": "reasoning",
        "prompt": "Find the exact remainder when 7^100 + 11^100 is divided by 13. Put the final integer inside \\boxed{...}.",
        "grader": "boxed_int",
        "expected": 12,
    },
    {
        "id": "reason_logic",
        "domain": "reasoning",
        "prompt": (
            "Five people (Alice, Bob, Carol, Dave, Eve) sit in seats 1 to 5 left to right.\n"
            "1. Alice is not at either end.\n"
            "2. Bob sits immediately to the left of Carol.\n"
            "3. Dave sits in seat 5.\n"
            "4. Eve sits somewhere to the left of Alice.\n"
            "5. Alice does not sit next to Dave.\n"
            "Who sits in seat 3? Put ONLY the single name inside \\boxed{...}."
        ),
        "grader": "boxed_str",
        "expected": "bob",
    },
    {
        "id": "reason_comb",
        "domain": "reasoning",
        "prompt": "How many 4-digit positive integers have strictly increasing digits from left to right? Put the final integer inside \\boxed{...}.",
        "grader": "boxed_int",
        "expected": 126,
    },
    # --- PYTHON ---
    {
        "id": "py_merge",
        "domain": "python",
        "prompt": "Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs and returns a sorted list of lists. Output only a ```python``` block.",
        "grader": "python",
        "test_code": (
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge_intervals([]) == []\n"
        ),
    },
    {
        "id": "py_lru",
        "domain": "python",
        "prompt": "Write a Python class `LRUCache(capacity: int)` with `get(key) -> int` (-1 if missing) and `put(key, value)` in O(1). Output only a ```python``` block.",
        "grader": "python",
        "test_code": (
            "c = LRUCache(2)\n"
            "c.put(1, 10); c.put(2, 20)\n"
            "assert c.get(1) == 10\n"
            "c.put(3, 30)\n"
            "assert c.get(2) == -1\n"
            "assert c.get(3) == 30\n"
        ),
    },
    # --- C ---
    {
        "id": "c_bsearch",
        "domain": "c",
        "prompt": "Write a C function `int bsearch_first(const int *arr, int n, int target)` returning the smallest index i where arr[i] == target, or -1 if absent. Output only a ```c``` block (no main).",
        "grader": "c",
        "test_code": (
            "#include <assert.h>\n"
            "int main(void) {\n"
            "  int a[] = {1, 2, 2, 2, 5};\n"
            "  assert(bsearch_first(a, 5, 2) == 1);\n"
            "  assert(bsearch_first(a, 5, 5) == 4);\n"
            "  assert(bsearch_first(a, 5, 3) == -1);\n"
            "  assert(bsearch_first(a, 0, 1) == -1);\n"
            "  return 0;\n"
            "}\n"
        ),
    },
    {
        "id": "c_revwords",
        "domain": "c",
        "prompt": "Write a C function `void reverse_str(char *s)` that reverses a NUL-terminated string in place. Output only a ```c``` block (no main).",
        "grader": "c",
        "test_code": (
            "#include <assert.h>\n"
            "#include <string.h>\n"
            "int main(void) {\n"
            "  char s1[] = \"abcdef\"; reverse_str(s1); assert(strcmp(s1, \"fedcba\") == 0);\n"
            "  char s2[] = \"\"; reverse_str(s2); assert(strcmp(s2, \"\") == 0);\n"
            "  return 0;\n"
            "}\n"
        ),
    },
    # --- PORTUGUESE ---
    {
        "id": "pt_math",
        "domain": "pt",
        "prompt": "Em português com acentuação correta: se um trem percorre 180 km em 2 horas e 30 minutos, qual é a velocidade média em km/h? Explique brevemente e coloque apenas o número inteiro em \\boxed{...}.",
        "grader": "boxed_int",
        "expected": 72,
    },
    {
        "id": "pt_code",
        "domain": "pt",
        "prompt": "Escreva em português uma breve explicação com acentuação correta e uma função Python `contar_vogais(texto)` que conta vogais (incluindo á, é, í, ó, ú, ã, õ, â, ê, ô). Inclua um bloco ```python```.",
        "grader": "python",
        "test_code": (
            "assert contar_vogais('coração') == 4\n"
            "assert contar_vogais('Python') == 1\n"
        ),
    },
]


def build_calibration_file(
    langs: list[str],
    domains: list[str],
    out_path: str,
    custom_ppl_file: str | None = None,
) -> str:
    if custom_ppl_file:
        return custom_ppl_file
    parts: list[str] = []
    lang_set = {l.lower() for l in langs}
    dom_set = {d.lower() for d in domains}
    if "en" in lang_set or not lang_set:
        parts.append(open(os.path.join(DATA_DIR, "en.txt"), encoding="utf-8").read())
    if "pt" in lang_set:
        parts.append(open(os.path.join(DATA_DIR, "pt.txt"), encoding="utf-8").read())
    if dom_set & {"python", "c", "code"}:
        parts.append(open(os.path.join(DATA_DIR, "code.txt"), encoding="utf-8").read())
    if not parts:
        parts.append(open(os.path.join(DATA_DIR, "en.txt"), encoding="utf-8").read())

    # Repeat slices cleanly so llama-perplexity has >= 16 chunks of 512 tokens
    base = "\n\n".join(parts)
    text = (base + "\n\n") * 4
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def load_eval_tasks(
    domains: list[str],
    langs: list[str],
    eval_file: str | None = None,
    quick: bool = False,
) -> list[dict[str, Any]]:
    active = {d.lower() for d in domains}
    if "pt" in {l.lower() for l in langs}:
        active.add("pt")

    selected: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    for t in BUILTIN_TASKS:
        if t["domain"] in active:
            if quick and t["domain"] in seen_domains:
                continue
            selected.append(dict(t))
            seen_domains.add(t["domain"])

    if eval_file:
        with open(eval_file, encoding="utf-8") as f:
            custom = json.load(f)
        for t in custom:
            selected.append(dict(t))
    return selected


def _extract_code_block(text: str, lang: str) -> str:
    m = re.findall(rf"```{lang}\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    m2 = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m2:
        return m2[-1].strip()
    return text.strip()


def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return m[-1].strip() if m else None


def _rlimit_preexec(timeout_s: int):
    def _set():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 1, timeout_s + 1))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    return _set


def grade_response(task: dict[str, Any], text: str, timeout_s: int = 10) -> tuple[bool, str]:
    grader = task["grader"]
    if grader == "boxed_int":
        val = _extract_boxed(text)
        if val is None:
            return False, "missing \\boxed{...}"
        try:
            got = int(re.sub(r"[^\d-]", "", val))
            return (got == int(task["expected"]), f"got {got}, expected {task['expected']}")
        except ValueError:
            return False, f"non-integer boxed value: {val}"

    if grader == "boxed_str":
        val = _extract_boxed(text)
        if val is None:
            return False, "missing \\boxed{...}"
        ok = val.strip().lower() == str(task["expected"]).strip().lower()
        return ok, f"got '{val}', expected '{task['expected']}'"

    if grader == "exact":
        ok = str(task["expected"]).strip() in text.strip()
        return ok, "exact match" if ok else "expected string not found"

    if grader == "regex":
        ok = bool(re.search(str(task["expected"]), text, flags=re.MULTILINE))
        return ok, "regex matched" if ok else "regex did not match"

    if grader == "python":
        code = _extract_code_block(text, "python") + "\n\n" + task.get("test_code", "")
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sol.py")
            with open(p, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                r = subprocess.run(
                    ["python3", p],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=_rlimit_preexec(timeout_s),
                )
                return (r.returncode == 0, "ok" if r.returncode == 0 else r.stderr.strip()[:200])
            except subprocess.TimeoutExpired:
                return False, f"timeout after {timeout_s}s"

    if grader == "c":
        code = _extract_code_block(text, "c") + "\n\n" + task.get("test_code", "")
        with tempfile.TemporaryDirectory() as td:
            src_p = os.path.join(td, "sol.c")
            bin_p = os.path.join(td, "sol")
            with open(src_p, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                c = subprocess.run(
                    ["gcc", "-O2", "-Wall", "-Wextra", "-Werror", src_p, "-o", bin_p],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if c.returncode != 0:
                    return False, "compile error: " + c.stderr.strip()[:200]
                r = subprocess.run(
                    [bin_p],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=_rlimit_preexec(timeout_s),
                )
                return (r.returncode == 0, "ok" if r.returncode == 0 else r.stderr.strip()[:200])
            except subprocess.TimeoutExpired:
                return False, f"timeout after {timeout_s}s"

    return False, f"unknown grader: {grader}"


def run_llama_bench(
    tc: ToolchainPaths,
    gguf_path: str,
    ctk: str = "f16",
    ctv: str = "f16",
    ubatch: int = 256,
    depth: int = 0,
    env_vars: dict[str, str] | None = None,
) -> tuple[float, float]:
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    cmd = [
        tc.llama_bench, "-m", gguf_path, "-fa", "1", "-ngl", "99",
        "-ctk", ctk, "-ctv", ctv, "-ub", str(ubatch),
        "-p", "512", "-n", "128", "-r", "3", "-o", "json",
    ]
    if depth > 0:
        cmd.extend(["-d", str(depth)])
    out = subprocess.check_output(cmd, text=True, env=env, stderr=subprocess.DEVNULL)
    rows = json.loads(out)
    pp_tps, tg_tps = 0.0, 0.0
    for r in rows:
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            pp_tps = float(r.get("avg_ts", 0.0))
        elif r.get("n_gen", 0) > 0:
            tg_tps = float(r.get("avg_ts", 0.0))
    return pp_tps, tg_tps


def run_perplexity(
    tc: ToolchainPaths,
    gguf_path: str,
    calib_file: str,
    ctk: str = "f16",
    ctv: str = "f16",
    chunks: int = 16,
    env_vars: dict[str, str] | None = None,
) -> float:
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    cmd = [
        tc.llama_perplexity, "-m", gguf_path, "-f", calib_file,
        "-fa", "on", "-ngl", "99", "-ctk", ctk, "-ctv", ctv,
        "-c", "512", "--chunks", str(chunks),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = res.stdout + "\n" + res.stderr
    m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", combined)
    if m:
        return float(m[-1])
    m2 = re.findall(r"\[(\d+)\]([0-9.]+)", combined)
    if m2:
        return float(m2[-1][1])
    raise RuntimeError(f"Failed to parse perplexity output: {combined[-300:]}")


def run_full_eval(
    tc: ToolchainPaths,
    gguf_path: str,
    calib_file: str,
    domains: list[str],
    langs: list[str],
    ctk: str = "f16",
    ctv: str = "f16",
    ubatch: int = 256,
    context: int = 8192,
    top_k: int = 40,
    vram_mb: int = 0,
    eval_file: str | None = None,
    quick: bool = False,
    env_vars: dict[str, str] | None = None,
    port: int = 18089,
) -> EvalMetrics:
    pp_tps, tg_tps = run_llama_bench(tc, gguf_path, ctk=ctk, ctv=ctv, ubatch=ubatch, env_vars=env_vars)
    ppl = run_perplexity(tc, gguf_path, calib_file, ctk=ctk, ctv=ctv, chunks=12 if quick else 20, env_vars=env_vars)

    tasks = load_eval_tasks(domains, langs, eval_file=eval_file, quick=quick)
    if not tasks:
        return EvalMetrics(pp_tps, tg_tps, ppl, 0, 0, vram_mb)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    srv_cmd = [
        tc.llama_server, "-m", gguf_path, "--host", "127.0.0.1", "--port", str(port),
        "-fa", "on", "-ngl", "99", "-c", str(min(context, 16384)),
        "-b", "512", "-ub", str(ubatch), "-ctk", ctk, "-ctv", ctv,
        "-np", "1", "--cache-ram", "0", "--top-k", str(top_k), "--jinja",
    ]
    proc = subprocess.Popen(srv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    passed = 0
    degenerate = False
    details: dict[str, Any] = {}
    try:
        # Wait for /health
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited prematurely during eval")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)

        max_toks = 1024 if quick else 2048
        for t in tasks:
            payload = json.dumps({
                "messages": [{"role": "user", "content": t["prompt"]}],
                "temperature": 0.0,
                "max_tokens": max_toks,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"].get("content", "") or ""
            if len(content.strip()) == 0 or (len(content) > 200 and len(set(content[-200:])) < 6):
                degenerate = True
            ok, note = grade_response(t, content)
            if ok:
                passed += 1
            details[t["id"]] = {"passed": ok, "note": note}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return EvalMetrics(pp_tps, tg_tps, ppl, passed, len(tasks), vram_mb, degenerate=degenerate, details=details)
