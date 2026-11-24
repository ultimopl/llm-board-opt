# llm-board-opt

Hardware-aware, closed-loop optimizer for running GGUF models with `llama.cpp` on a single local Linux board.

It probes the board, builds a board-native `llama.cpp`, measures every GGML quant format on the real hardware, then
greedily tries GGUF changes (re-quantization, vocab pruning, MoE expert pruning) and runtime settings (KV cache types,
ubatch). A change is kept only when a quality gate says it is faster or saves memory without hurting perplexity or
task evals. The result is an optimized `.gguf`, a `serve.sh`, and optionally a running `llama-server`.

Developed and tested on an ASRock BC-250 (AMD Cyan Skillfish, `RADV GFX1013`, 16 GiB unified memory, Vulkan). The
build step also supports ROCm, CUDA and CPU-only boards.

## Requirements

- Python ≥ 3.10, `numpy`
- `git`, `cmake`, `g++`/`gcc` (to build `llama.cpp` and the native helpers; `gcc` also grades C eval tasks)
- `vulkaninfo`, `rocminfo` or `nvidia-smi` for GPU detection (auto-detected; override with `--backend`)
- `systemd-run --user` for `--serve` / `board-opt serve`

## Install

```bash
pip install -e . --no-build-isolation --no-deps
```

## Commands

| Command | What it does |
|---|---|
| `board-opt probe` | Detect CPU `-march`, GPU backend and usable VRAM; print JSON. |
| `board-opt budget --model M.gguf` | Parse the GGUF and list the `(ctk, ctv, ubatch, context)` configs that fit in VRAM. |
| `board-opt inspect --model M.gguf` | Per-layer quant types, prunable MoE experts, vocab scripts, and the expected gain/loss of each change. Read-only. |
| `board-opt bench-quants` | Build the toolchain and benchmark every GGML quant format on this board (decode/prefill speed, bandwidth, conversion speed, RMSE/CosSim). |
| `board-opt run --model M.gguf` | The full pipeline below. |
| `board-opt serve --work-dir DIR` | Launch a `serve.sh` produced by an earlier `run`. |

Run `board-opt <command> -h` for all flags.

## Quick start

```bash
board-opt probe
board-opt bench-quants
board-opt budget  --model ~/models/gemma-4-26B-A4B-it-UD-IQ3_S.gguf --context 32768
board-opt inspect --model ~/models/gemma-4-26B-A4B-it-UD-IQ3_S.gguf

board-opt run \
  --model ~/models/gemma-4-26B-A4B-it-UD-IQ3_S.gguf \
  --langs en,pt \
  --domains reasoning,python,c \
  --context 32768 \
  --serve --port 8080 -v
```

## Pipeline (`board-opt run`)

| Stage | Step |
|---|---|
| 0 | **Probe**: CPU ISA, backend, usable VRAM (total minus a reserve of max(1536 MiB, 10%), or `--vram-reserve-mb`). |
| 1 | **Budget**: parse GGUF v3 (including hybrid/SWA layouts such as `qwen35`, `gemma4`) and require `weights + KV + scratch ≤ usable VRAM`. Aborts if the weights alone don't fit. |
| 2 | **Build**: static, `-march=native` `llama.cpp` plus native helpers (`quant_bench`, `reap_trace`, `ggml_conv`); then the per-quant benchmark. |
| 3 | **Baseline**: `llama-bench`, `llama-perplexity` on a calibration file built from `--langs`/`--domains`, and graded domain evals. |
| 4 | **Runtime sweep**: try budget-valid `ctk/ctv/ubatch` configs through the gate; on Vulkan also tunes `GGML_VK_MAX_NODES_PER_SUBMIT`. |
| 5 | **GGUF optimization**, each step gated: fix non-native tensor types → re-quantize attention/output to the best measured format → prune GPT-2 BPE vocab to the scripts of `--langs` → reduce active MoE experts → REAP expert pruning (`--prune-experts N`) → optional recovery hook. |
| 6 | **Package**: `model-optimized.gguf` + `serve.sh`; with `--serve`, start `llama-server` via `systemd-run --user`, wait for `/health`, and send a smoke chat request. |

Vocab pruning only applies to `gpt2` byte-level BPE tokenizers; other tokenizers are skipped and the skip is recorded.

### Quality gate

A candidate is rejected if it produces degenerate output, passes fewer eval tasks than the baseline minus
`--allow-eval-drop` (default 0), or raises perplexity by more than `--max-ppl-delta` (default 3%). Otherwise it is
accepted if decode speed improves by at least `--min-tg-gain` (default 1.5%), or, when the model is VRAM-constrained,
if it saves at least `--min-vram-save-mib` (default 100 MiB).

### Evals

Built-in tasks cover the domains `reasoning`, `python`, `c` and `pt` (added automatically when `pt` is in `--langs`).
Python and C answers are executed (C is compiled with `gcc -Wall -Wextra -Werror`) under resource limits.
Add your own with `--eval-file tasks.json`, a JSON list of tasks:

```json
[
  {"id": "my_task", "domain": "reasoning", "prompt": "... Put the answer in \\boxed{...}.", "grader": "boxed_int", "expected": 42}
]
```

Graders: `boxed_int`, `boxed_str`, `exact` (substring), `regex`, `python` / `c` (run the extracted code block with
`test_code` appended). Use `--ppl-file` to supply your own perplexity text.

## Files and resume

- **llama.cpp source**: `--llama-src`, else `~/src/llama-mainline`, else a shallow clone into `~/.cache/board-opt/llama.cpp`.
- **Build**: `~/.cache/board-opt/llama-native-<hardware fingerprint>/` (an existing `~/llama-bc250-native` is reused unless `--rebuild`).
- **Work dir**: `--work-dir` (default `./work`) `/<model-slug>/` holds one JSON per stage (`hardware`, `quant_bench`,
  `baseline`, `tune`, `optimize`, `final`, …), `calibration.txt`, `model-optimized.gguf` and `serve.sh`.
  A rerun resumes from those JSONs; pass `--no-resume` to start over. Intermediate GGUFs are deleted unless
  `--keep-intermediates`.
- `inspect --quant-bench-json work/<slug>/quant_bench.json` reuses the measured table for exact speed/loss estimates.

## One GPU job at a time

Before any GPU work, `run`, `bench-quants` and `serve` check for running `llama-server`/`llama-bench`/
`llama-perplexity`/`llama-cli` processes and refuse to start alongside them (the board's unified memory can't
hold two models). The server unit is named `board-opt-<slug>` unless `--unit-name` is given.

## Extending

`board_opt/recovery.py` defines `RecoveryHook`, a hook for post-quantization recovery (e.g. evolution strategies or
QAT). It receives the current GGUF and an eval function and returns a new GGUF path, which then goes through the gate.
The default is a no-op.

## Tests

```bash
pytest -v
```

The tests run offline and don't need a GPU or a model.

## License

Apache-2.0, see [LICENSE](LICENSE).
