// reap_trace.cpp — Exact REAP (Router-weighted Expert Activation Pruning) calibrator
// for Inkling-2-Mini (inkling2_loop) using llama.cpp's cb_eval callback.
// Aggregates sum(|weight| * ||expert_out||_2) per physical layer (0..23) and expert (0..127).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static constexpr int N_PHYS = 24;
static constexpr int N_EXP  = 128;

struct ReapState {
    double reap_score[N_PHYS][N_EXP] = {};
    double weight_sum[N_PHYS][N_EXP] = {};
    uint64_t hit_count[N_PHYS][N_EXP] = {};
    bool printed_names = false;

    // Temporary per-virtual-layer buffers during a single graph eval
    int cur_vlayer_topk = -1;
    std::vector<int32_t> topk_ids;
    int topk_k = 0;
    int topk_n = 0;

    int cur_vlayer_w = -1;
    std::vector<float> weights;
};

static int vlayer_to_phys(int il) {
    // Inkling-2-Mini: 24 physical layers, loop.start=6, loop.end=18, loop.count=3 -> 48 virtual layers
    if (il < 6) return il;
    if (il < 42) return 6 + ((il - 6) % 12);
    int p = 18 + (il - 42);
    return (p >= 0 && p < N_PHYS) ? p : -1;
}

static int parse_layer_suffix(const char * name, const char * prefix) {
    size_t plen = std::strlen(prefix);
    if (std::strncmp(name, prefix, plen) != 0) return -1;
    const char * p = name + plen;
    if (*p < '0' || *p > '9') return -1;
    return std::atoi(p);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<ReapState *>(user_data);
    const char * name = ggml_get_name(t);
    if (!name || !name[0]) return true;

    bool is_topk = (std::strncmp(name, "ffn_moe_argsort-", 16) == 0);
    bool is_w    = (std::strncmp(name, "ffn_moe_weights_scaled-", 23) == 0) ||
                   (std::strncmp(name, "ffn_moe_weights-", 16) == 0);
    bool is_down = (std::strncmp(name, "ffn_moe_down-", 13) == 0);

    if (ask) {
        return is_topk || is_w || is_down;
    }

    if (is_topk) {
        int il = parse_layer_suffix(name, "ffn_moe_argsort-");
        if (il < 0) return true;
        int n_exp = (int)t->ne[0];
        int n = (int)t->ne[1];
        int k = 8;
        std::vector<int32_t> full((size_t)n_exp * n);
        ggml_backend_tensor_get(t, full.data(), 0, full.size() * sizeof(int32_t));
        st->topk_ids.resize((size_t)k * n);
        for (int tok = 0; tok < n; tok++) {
            for (int j = 0; j < k; j++) {
                st->topk_ids[tok * k + j] = full[tok * n_exp + j];
            }
        }
        st->cur_vlayer_topk = il;
        st->topk_k = k;
        st->topk_n = n;
        int phys = vlayer_to_phys(il);
        if (phys >= 0) {
            for (int tok = 0; tok < n; tok++) {
                for (int j = 0; j < k; j++) {
                    int e = st->topk_ids[tok * k + j];
                    if (e >= 0 && e < N_EXP) {
                        st->hit_count[phys][e]++;
                    }
                }
            }
        }
    } else if (is_w) {
        int il = parse_layer_suffix(name, "ffn_moe_weights_scaled-");
        if (il < 0) il = parse_layer_suffix(name, "ffn_moe_weights-");
        if (il < 0 || il != st->cur_vlayer_topk) return true;
        size_t ne = (size_t)ggml_nelements(t);
        if (t->type == GGML_TYPE_F32 && ne == st->topk_ids.size() && ggml_is_contiguous(t)) {
            st->weights.resize(ne);
            ggml_backend_tensor_get(t, st->weights.data(), 0, ne * sizeof(float));
            st->cur_vlayer_w = il;
            int phys = vlayer_to_phys(il);
            if (phys >= 0) {
                for (size_t idx = 0; idx < ne; idx++) {
                    int e = st->topk_ids[idx];
                    if (e >= 0 && e < N_EXP) {
                        st->weight_sum[phys][e] += std::fabs((double)st->weights[idx]);
                    }
                }
            }
        }
    } else if (is_down) {
        int il = parse_layer_suffix(name, "ffn_moe_down-");
        if (il < 0 || il != st->cur_vlayer_topk) return true;
        int phys = vlayer_to_phys(il);
        if (phys < 0 || t->type != GGML_TYPE_F32) return true;
        // shape: [n_embd, k, n_tokens]
        int d = (int)t->ne[0];
        int k = (int)t->ne[1];
        int n = (int)t->ne[2];
        if (k != st->topk_k || n != st->topk_n) return true;
        std::vector<float> buf((size_t)d * k * n);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        bool have_w = (st->cur_vlayer_w == il && st->weights.size() == (size_t)k * n);
        for (int tok = 0; tok < n; tok++) {
            for (int j = 0; j < k; j++) {
                int e = st->topk_ids[tok * k + j];
                if (e < 0 || e >= N_EXP) continue;
                const float * vec = buf.data() + ((size_t)(tok * k + j) * d);
                double sq = 0.0;
                for (int c = 0; c < d; c++) sq += (double)vec[c] * (double)vec[c];
                double norm = std::sqrt(sq);
                double w = have_w ? std::fabs((double)st->weights[tok * k + j]) : 1.0;
                st->reap_score[phys][e] += w * norm;
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf CALIB.txt OUT.json [max_tokens=8192] [chunk=512]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * calib_path = argv[2];
    const char * out_path   = argv[3];
    int max_tokens = (argc > 4) ? std::atoi(argv[4]) : 8192;
    int chunk_sz   = (argc > 5) ? std::atoi(argv[5]) : 512;

    std::ifstream ifs(calib_path);
    if (!ifs) { std::fprintf(stderr, "Cannot open %s\n", calib_path); return 1; }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string text = ss.str();

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "Failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 16);
    int n_tok = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), false, true);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), false, true);
    }
    if (n_tok > max_tokens) n_tok = max_tokens;
    tokens.resize(n_tok);
    std::fprintf(stderr, "Tokenized %d calibration tokens (chunk=%d)\n", n_tok, chunk_sz);

    ReapState st;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = chunk_sz;
    cparams.n_batch = chunk_sz;
    cparams.n_ubatch = chunk_sz;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "Failed to create context\n"); return 1; }

    int n_chunks = (n_tok + chunk_sz - 1) / chunk_sz;
    for (int ci = 0; ci < n_chunks; ci++) {
        int start = ci * chunk_sz;
        int len = std::min(chunk_sz, n_tok - start);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(tokens.data() + start, len);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed at chunk %d\n", ci);
            return 1;
        }
        std::fprintf(stderr, "Chunk %d/%d done (%d tokens)\n", ci + 1, n_chunks, len);
    }

    // Write JSON
    FILE * f = std::fopen(out_path, "w");
    std::fprintf(f, "{\n  \"n_tokens\": %d,\n  \"reap\": [\n", n_tok);
    for (int l = 0; l < N_PHYS; l++) {
        std::fprintf(f, "    [");
        for (int e = 0; e < N_EXP; e++) {
            std::fprintf(f, "%.6f%s", st.reap_score[l][e], (e + 1 < N_EXP) ? "," : "");
        }
        std::fprintf(f, "]%s\n", (l + 1 < N_PHYS) ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"weight_sum\": [\n");
    for (int l = 0; l < N_PHYS; l++) {
        std::fprintf(f, "    [");
        for (int e = 0; e < N_EXP; e++) {
            std::fprintf(f, "%.6f%s", st.weight_sum[l][e], (e + 1 < N_EXP) ? "," : "");
        }
        std::fprintf(f, "]%s\n", (l + 1 < N_PHYS) ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"hits\": [\n");
    for (int l = 0; l < N_PHYS; l++) {
        std::fprintf(f, "    [");
        for (int e = 0; e < N_EXP; e++) {
            std::fprintf(f, "%llu%s", (unsigned long long)st.hit_count[l][e], (e + 1 < N_EXP) ? "," : "");
        }
        std::fprintf(f, "]%s\n", (l + 1 < N_PHYS) ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    std::fprintf(stderr, "Wrote REAP scores to %s\n", out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
