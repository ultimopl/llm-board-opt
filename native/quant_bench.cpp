#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct BenchRes {
    const char * name;
    int type_id;
    bool supported;
    double bpw;
    double tg_ms;
    double tg_gws;
    double tg_gbps;
    double pp_gws;
    double conv_ms_per_m;
    double rmse;
    double cos_sim;
};

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static void compute_loss(const std::vector<float> & ref, const std::vector<float> & got,
                         double & rmse, double & cos_sim) {
    double se = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = double(got[i]) - double(ref[i]);
        se += d * d;
        dot += double(ref[i]) * double(got[i]);
        nr += double(ref[i]) * double(ref[i]);
        ng += double(got[i]) * double(got[i]);
    }
    rmse = std::sqrt(se / double(ref.size()));
    cos_sim = (nr > 0.0 && ng > 0.0) ? (dot / (std::sqrt(nr) * std::sqrt(ng))) : 0.0;
}

static double bench_mul_mat(ggml_backend_t backend, ggml_type wtype, const void * wdata, size_t wbytes,
                            const float * xdata, int64_t M, int64_t K, int64_t N,
                            int warmup, int iters, std::vector<float> * out_y) {
    struct ggml_init_params params = {
        /* mem_size */ 16 * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!ggml_backend_dev_supports_op(dev, Y)) {
        ggml_free(ctx);
        return -1.0;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return -1.0;
    }

    ggml_backend_tensor_set(W, wdata, 0, wbytes);
    ggml_backend_tensor_set(X, xdata, 0, size_t(K) * size_t(N) * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return -1.0;
        }
    }
    ggml_backend_synchronize(backend);

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        times[i] = now_ms() - t0;
    }
    std::sort(times.begin(), times.end());

    if (out_y) {
        out_y->resize(size_t(M) * size_t(N));
        ggml_backend_tensor_get(Y, out_y->data(), 0, out_y->size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return times[iters / 2];
}

int main(int argc, char ** argv) {
    int64_t M = (argc > 1) ? std::atoll(argv[1]) : 4096;
    int64_t K = (argc > 2) ? std::atoll(argv[2]) : 4096;
    int64_t N_pp = 32;

    ggml_backend_t backend = ggml_backend_init_best();
    if (!backend) {
        fprintf(stderr, "Failed to initialize ggml backend\n");
        return 1;
    }
    const char * bname = ggml_backend_name(backend);

    // Deterministic Gaussian-like weights & activations
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<float> W_f32(size_t(M) * size_t(K));
    for (auto & v : W_f32) v = dist(rng);

    std::vector<float> X_tg(static_cast<size_t>(K));
    for (auto & v : X_tg) v = dist(rng) * 10.0f;
    std::vector<float> X_pp(static_cast<size_t>(K) * static_cast<size_t>(N_pp));
    for (auto & v : X_pp) v = dist(rng) * 10.0f;

    // Exact CPU FP32 reference for N=1 matvec
    std::vector<float> Y_ref(size_t(M), 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        double s = 0.0;
        const float * row = W_f32.data() + size_t(m) * size_t(K);
        for (int64_t k = 0; k < K; ++k) s += double(row[k]) * double(X_tg[k]);
        Y_ref[m] = float(s);
    }

    const ggml_type candidates[] = {
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q4_0,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_Q3_K,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_Q2_K,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_TQ2_0,
        GGML_TYPE_TQ1_0,
        GGML_TYPE_Q2_0,
        GGML_TYPE_MXFP4,
        GGML_TYPE_NVFP4,
    };

    std::vector<BenchRes> results;
    for (ggml_type t : candidates) {
        const char * tname = ggml_type_name(t);
        if (!tname) continue;
        size_t row_sz = ggml_row_size(t, K);
        if (row_sz == 0) continue;
        size_t total_bytes = row_sz * size_t(M);
        double bpw = (double(total_bytes) * 8.0) / (double(M) * double(K));

        // Measure quantization / conversion speed
        std::vector<uint8_t> qbuf(total_bytes);
        double c0 = now_ms();
        size_t written = 0;
        if (t == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(W_f32.data(), (ggml_fp16_t *)qbuf.data(), M * K);
            written = total_bytes;
        } else if (ggml_get_type_traits_cpu(t) && ggml_get_type_traits_cpu(t)->from_float) {
            written = ggml_quantize_chunk(t, W_f32.data(), qbuf.data(), 0, M, K, nullptr);
        } else {
            results.push_back({tname, (int)t, false, bpw, 0, 0, 0, 0, 0, 0, 0});
            continue;
        }
        double conv_ms = now_ms() - c0;
        if (written == 0) {
            results.push_back({tname, (int)t, false, bpw, 0, 0, 0, 0, 0, 0, 0});
            continue;
        }

        std::vector<float> y_got;
        double tg_ms = bench_mul_mat(backend, t, qbuf.data(), total_bytes, X_tg.data(), M, K, 1, 5, 15, &y_got);
        if (tg_ms <= 0.0) {
            results.push_back({tname, (int)t, false, bpw, 0, 0, 0, 0, conv_ms / (double(M * K) / 1e6), 0, 0});
            continue;
        }
        double pp_ms = bench_mul_mat(backend, t, qbuf.data(), total_bytes, X_pp.data(), M, K, N_pp, 3, 8, nullptr);

        double rmse = 0.0, cos_sim = 0.0;
        compute_loss(Y_ref, y_got, rmse, cos_sim);

        double n_weights = double(M) * double(K);
        double tg_gws  = (n_weights / (tg_ms * 1e-3)) / 1e9;
        double tg_gbps = (double(total_bytes) / (tg_ms * 1e-3)) / 1e9;
        double pp_gws  = (pp_ms > 0.0) ? ((n_weights * double(N_pp) / (pp_ms * 1e-3)) / 1e9) : 0.0;
        double conv_per_m = conv_ms / (n_weights / 1e6);

        results.push_back({tname, (int)t, true, bpw, tg_ms, tg_gws, tg_gbps, pp_gws, conv_per_m, rmse, cos_sim});
    }

    printf("{\n  \"backend\": \"%s\",\n  \"M\": %lld,\n  \"K\": %lld,\n  \"results\": [\n",
           bname, (long long)M, (long long)K);
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & r = results[i];
        printf("    {\"name\": \"%s\", \"type_id\": %d, \"supported\": %s, \"bpw\": %.3f, "
               "\"tg_ms\": %.4f, \"tg_gws\": %.2f, \"tg_gbps\": %.2f, \"pp_gws\": %.2f, "
               "\"conv_ms_per_m\": %.3f, \"rmse\": %.6f, \"cos_sim\": %.6f}%s\n",
               r.name, r.type_id, r.supported ? "true" : "false", r.bpw,
               r.tg_ms, r.tg_gws, r.tg_gbps, r.pp_gws,
               r.conv_ms_per_m, r.rmse, r.cos_sim,
               (i + 1 < results.size()) ? "," : "");
    }
    printf("  ]\n}\n");

    ggml_backend_free(backend);
    return 0;
}
