# Hardware & Model Optimization Findings

Empirical discoveries made while building and testing `llm-board-opt` on **AMD BC-250 (`RADV GFX1013`, 16 GiB unified GDDR6, Vulkan, `fp16=1`, `bf16=0`, `fp4=0`, `int dot=0`)** and **NVIDIA Tesla T4 (CUDA `sm_75`)**.

---

## 1. SenseNova-Loop-9B: Closed-Form 2×2 Sinkhorn-Knopp (+112% / 2.12× Lossless Speedup)

### Root Cause
`SenseNova-Loop-9B` (`sensenova_loop`) sets `sensenova_loop.mhc.iterations = 20`. In `OpenSenseNova/llama.cpp` (`src/models/sensenova-loop.cpp`), each of the 96 manifold-constrained hyper-connection (mHC) blocks per token executes 20 sequential Sinkhorn-Knopp normalization iterations on a **2×2 matrix**, emitting 24 scalar GGML ops per iteration:

$$96\text{ blocks} \times 20\text{ iters} \times 24\text{ ops} = 46{,}080\text{ sequential scalar Vulkan kernels per token}$$

At ~3 µs per Vulkan dispatch/barrier, dispatch overhead alone consumed ~138 ms/token, capping decode at **7.4 tok/s** despite the model being only 5.41 GiB.

### Exact Mathematical Fix
For any positive 2×2 matrix $M = \begin{pmatrix} e^a & e^b \\ e^c & e^d \end{pmatrix}$, cross-ratio invariance under Sinkhorn row/column scaling gives the **exact** $\infty$-iteration doubly stochastic matrix $P = \begin{pmatrix} p & 1-p \\ 1-p & p \end{pmatrix}$ in closed form:

$$\frac{p^2}{(1-p)^2} = e^{a + d - b - c} \implies p = \sigma\!\left(\frac{a + d - b - c}{2}\right), \quad q = \sigma\!\left(-\frac{a + d - b - c}{2}\right)$$

Replacing the 20-iteration loop with 6 GGML ops (`add`, `add`, `sub`, `scale`, `sigmoid`, `neg`+`sigmoid`) cuts 46,080 scalar dispatches down to ~480:

| Build (`SenseNova-Loop-9B-Q4_K_M`, 5.41 GiB, BC-250 Vulkan) | `pp512` (tok/s) | `tg128` (tok/s) | Perplexity (20 chunks) |
|---|---:|---:|---:|
| Stock `OpenSenseNova/llama.cpp` (20 Sinkhorn iters) | 212.70 | 7.40 | 5.264 |
| **Patched closed-form 2×2 Sinkhorn** | **247.42 (+16.3%)** | **15.71 (+112% / 2.12×)** | **4.946** |

---

## 2. Empirical GGML Quantization Profile on AMD BC-250 (`4096 × 4096`, Vulkan)

Measured via `board-opt bench-quants` (decode `N=1`, prefill `N=32`, CPU conversion speed, and matvec RMSE/CosSim vs FP32):

| Quant | BPW | `tg` (GW/s) | `tg` (GB/s) | `pp` (GW/s) | Conv (ms/M) | RMSE | CosSim |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iq4_nl` | 4.50 | **212.6** | 119.6 | 1168.0 | 33.96 | 0.12124 | 0.997197 |
| `q4_K` | 4.50 | **212.5** | 119.5 | 1113.5 | 66.28 | 0.11367 | **0.997535** |
| `q4_0` | 4.50 | **211.6** | 119.0 | **1278.8** | **1.04** | 0.13847 | 0.996358 |
| `q6_K` | 6.56 | **208.6** | 171.1 | 921.4 | 34.55 | **0.02808** | **0.999850** |
| `q8_0` | 8.50 | 169.4 | 180.0 | 1216.6 | 4.62 | 0.00835 | 0.999987 |
| `q5_K` | 5.50 | 169.0 | 116.2 | 1068.8 | 55.49 | 0.05773 | 0.999365 |
| `q5_0` | 5.50 | 140.7 | 96.8 | 1114.2 | 1.36 | 0.06810 | 0.999121 |
| `nvfp4` | 4.50 | 140.5 | 79.0 | 1163.1 | 16.55 | 0.15498 | 0.995414 |
| `tq2_0` | 2.06 | 211.6 | 54.6 | 1065.0 | 4.17 | 1.28940 | 0.705493 |
| `tq1_0` | 1.69 | 140.5 | 29.6 | 556.6 | 4.38 | 1.28940 | 0.705493 |
| `f16` | 16.00 | 94.1 | 188.3 | 925.6 | 2.38 | 0.00033 | 1.000000 |

**Key takeaways:**
- **`q6_K` beats both `q5_K` and `q5_0` in speed AND accuracy** on BC-250 (`208.6` vs `169.0` / `140.7 GW/s`, 2× lower RMSE).
- **`nvfp4` is 34% slower than `q4_0`/`q4_K`** (and higher RMSE) on boards without native FP4 hardware (`fp4=0`).
- **Base-3 ternary (`tq1_0`) is 34% slower on decode and 2× slower on prefill** than 2-bit-slot ternary (`tq2_0` / `q2_0`).

---

## 3. Custom Vulkan Ternary Compute Micro-Benchmark (BC-250)

Comparing ternary matvec strategies against `Q4_0` on identical weights (`M=16384, K=4096`, Ternary `16 MiB` vs `Q4_0` `36 MiB`):

| Kernel | Median (ms) | Eff GB/s | GW/s | vs Best `Q4_0` |
|---|---:|---:|---:|---:|
| **`ternary_2bp_nobar` (2-bitplane mask + int add, barrier-free)** | **0.163** | 102.7 | **410.7** | **2.13×** |
| `ternary_swar_i16` (base-3 → packed `i16vec2`) | 0.315 | 53.3 | 213.3 | 1.10× |
| `ternary_unpack_fp32` (base-3 → `fp32` FMA) | 0.326 | 51.5 | 205.8 | 1.06× |
| `q4_0_fp32` (tiled 4.5 bpw baseline) | 0.347 | 108.7 | 193.3 | 1.00× |
| `ternary_lut_i16x2` (81-entry LDS LUT + 1 add) | 0.380 | 44.1 | 176.6 | 0.91× |

- **2-bitplane mask-and-add (`acc += (x & pos) - (x & neg)`)** hit **2.13× faster than `Q4_0`** (nearly the 2.25× memory ratio) with zero multiplies, zero divisions, and zero LDS lookups.
- **LDS LUT lost by ~24% on BC-250** due to Wave32 LDS bank conflicts and per-tile `barrier()` overhead.

---

## 4. Ternary-Bonsai-2-9B (`Q2_0`) vs. Qwen3.5-9B (`Q5_K_M`) on BC-250

Both models share the exact same `qwen35 9B` (8.95B) architecture:

| Model | Size | `pp512` (tok/s) | `tg128` (tok/s) | Effective Bandwidth |
|---|---:|---:|---:|---:|
| `Qwen3.5-9B-Q5_K_M.gguf` | 6.12 GiB | 512.09 | 51.96 | **~318 GiB/s** |
| `Ternary-Bonsai-2-9B-Q2_0.gguf` | 2.37 GiB *(2.58× smaller)* | **609.04 (+18.9%)** | **59.64 (+14.8%)** | **~141 GiB/s** |

**Why `Q2_0` left ~half its speed on the table:** `llama.cpp`'s fast `mul_mat_vec_q2_0_q8_1_f32` Vulkan shader requires `VK_KHR_shader_integer_dot_product` (`int dot=1`), which BC-250 lacks (`int dot=0`). It falls back to byte-by-byte loads in `dequant_funcs.glsl:147`.

---

## 5. End-to-End Model Optimizations (BC-250)

- **Inkling-2-Mini (11.1 GiB, `inkling2_loop`):**
  - `general.sampling.top_k = 0` on a 201k vocab caused an `O(V log V)` CPU full-sort on every token (~7–14 ms/tok); setting `top_k = 40` gave **+21.9% lossless live `tg`**.
  - Re-quantizing `F32` loop controllers + `Q8_0`/`Q6_K`/`Q5_K` attention/output + 20 `NVFP4` expert tensors → `Q4_0`, pruning vocab `200,058 → 113,778` (EN+PT+Code, -215.6 MB), lowering `expert_used_count` `8 → 6`, and native `-march=znver2` LTO compilation lifted `tg` from **32.96 → 43.78 t/s** (`llama-bench`) and **22.66 → 38.24 t/s (+68.8%)** live, while improving 12-task eval from **10/12 → 12/12**.
- **Gemma 4 26B-A4B (10.77 GiB, `gemma4`):**
  - Correctly modeling hybrid SWA (`5` full-attention KV layers + `25` SWA layers) unlocked 32K context in 13.9 GiB UMA.
  - Quantized KV (`q8_0/q4_0`, `q4_0/q4_0`) slowed `tg` by -3% on the small 5-layer KV cache and was automatically rejected in favor of `f16/f16`.
  - Re-quantizing `Q8_0`/`Q6_K` attention & output → `Q4_0` lifted `tg` **83.17 → 87.60 t/s (+5.3%)**, `pp` **544.25 → 566.00 t/s (+4.0%)**, saved **284 MiB VRAM**, and improved eval **6/10 → 7/10**.
- **NVIDIA Tesla T4 (Colab CUDA `sm_75`):**
  - Passing `-DCMAKE_CUDA_ARCHITECTURES=native` is essential to avoid compiling 6+ CUDA architectures on low-vCPU VMs.
  - Native C++ helpers (`quant_bench`, `ggml_conv`, `reap_trace`) must be built via CMake `target_link_libraries(... PRIVATE ggml)` rather than manual `g++ -l...` flags so CUDA device-link / `-lcublasLt` links cleanly.
