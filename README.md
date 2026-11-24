# llm-board-opt (v0.1.0)

**Hardware-aware, closed-loop LLM optimizer for `llama.cpp` on local Linux boards** (tested on AMD BC-250 `RADV GFX1013` 16 GiB UMA, works on Vulkan / CUDA / ROCm / CPU).

> **Status: v0.1.0** — Working end-to-end prototype with empirical quant profiling, VRAM/KV budgeting, quality-gated KV sweeps, GGUF re-quantization, GPT-2 BPE vocab pruning, MoE active/REAP expert pruning, and `systemd-run` server deployment. Actively evolving.

---

## Pipeline

1. **Stage 0 — Hardware Probe (`board-opt probe`)**: Detects CPU ISA, usable VRAM/UMA with safety reserve, GPU backend (`vulkan`/`rocm`/`cuda`/`cpu`), and preferred formats.
2. **Stage 1 — GGUF Inspector & VRAM/KV Budget (`board-opt budget`)**: Parses GGUF v3 headers (including hybrid/SWA architectures like `qwen35` and `gemma4`) and enforces `weights + KV(ctx, ctk, ctv) + scratch <= usable_vram`.
3. **Stage 2 — Native `llama.cpp` Build + Empirical Quant Benchmark (`board-opt bench-quants`)**: Compiles static board-native `llama.cpp` + C++ helpers (`ggml_conv`, `reap_trace`, `quant_bench`) and measures **actual decode GW/s, GDDR6 GB/s, prefill GW/s, conversion speed, and RMSE/CosSim loss** for every GGML quant format on the board.
4. **Stage 3 — Baseline Lock**: Measures `llama-bench`, offline `llama-perplexity`, and objective domain evals (`reasoning`, `python`, `c`, `pt`, or custom `--eval-file`).
5. **Stage 4 — KV & Runtime Permutation Sweep**: Screens budget-valid `(ctk, ctv, ubatch)` configs under a quality gate so broken KV quantizations are rejected automatically.
6. **Stage 5 — Greedy GGUF Optimization**: Applies metadata fixes, empirically chosen hardware-native tensor re-quantization, Latin-script GPT-2 BPE vocabulary pruning, and MoE active/REAP expert pruning — keeping only steps that pass the quality gate. Includes a `RecoveryHook` interface for Phase 2 post-quantization recovery (e.g. Evolution Strategies).
7. **Stage 6 — Server Packaging (`board-opt serve`)**: Emits `model-optimized.gguf`, `serve.sh`, and launches a `systemd-run --user` `llama-server` with `/health` + live smoke verification.

---

## Quick Start

```bash
pip install -e . --no-build-isolation --no-deps

# 1. Probe board hardware
board-opt probe

# 2. Benchmark all GGML quant formats on your board (speed, GB/s, conv speed, RMSE/CosSim)
board-opt bench-quants

# 3. Check VRAM/KV budget for a GGUF
board-opt budget --model ~/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ3_S.gguf --context 32768

# 4. Run full optimization pipeline + launch server on port 8080
board-opt run \
  --model ~/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ3_S.gguf \
  --langs en,pt \
  --domains reasoning,python,c \
  --context 32768 \
  --serve --port 8080 -v
```

## Testing

```bash
pytest -v
```
