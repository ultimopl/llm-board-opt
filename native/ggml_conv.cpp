// ggml_conv.cpp — Convert raw GGML tensor bytes from any ggml_type to another using libggml-base.so
// Usage: ggml_conv <src_type_id> <dst_type_id> <n_elements_per_row> <n_rows> <in.bin> <out.bin>
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "Usage: %s src_type dst_type ncols nrows in.bin out.bin\n", argv[0]);
        return 1;
    }
    ggml_type src_t = (ggml_type)std::atoi(argv[1]);
    ggml_type dst_t = (ggml_type)std::atoi(argv[2]);
    int64_t ncols   = std::atoll(argv[3]);
    int64_t nrows   = std::atoll(argv[4]);
    const char * in_path  = argv[5];
    const char * out_path = argv[6];

    size_t src_row_b = ggml_row_size(src_t, ncols);
    size_t dst_row_b = ggml_row_size(dst_t, ncols);

    const auto * src_traits = ggml_get_type_traits(src_t);
    if (src_t != GGML_TYPE_F32 && (!src_traits || !src_traits->to_float)) {
        std::fprintf(stderr, "No to_float for src_type=%d\n", (int)src_t);
        return 2;
    }

    FILE * fin  = std::fopen(in_path, "rb");
    FILE * fout = std::fopen(out_path, "wb");
    if (!fin || !fout) { std::fprintf(stderr, "Failed to open files\n"); return 3; }

    std::vector<uint8_t> src_buf(src_row_b);
    std::vector<float>   f32_buf((size_t)ncols);
    std::vector<uint8_t> dst_buf(dst_row_b);

    ggml_quantize_init(dst_t);

    for (int64_t r = 0; r < nrows; r++) {
        if (std::fread(src_buf.data(), 1, src_row_b, fin) != src_row_b) {
            std::fprintf(stderr, "Short read at row %lld\n", (long long)r);
            return 4;
        }
        if (src_t == GGML_TYPE_F32) {
            std::memcpy(f32_buf.data(), src_buf.data(), (size_t)ncols * sizeof(float));
        } else {
            src_traits->to_float(src_buf.data(), f32_buf.data(), ncols);
        }
        ggml_quantize_chunk(dst_t, f32_buf.data(), dst_buf.data(), 0, 1, ncols, nullptr);
        if (std::fwrite(dst_buf.data(), 1, dst_row_b, fout) != dst_row_b) {
            std::fprintf(stderr, "Short write at row %lld\n", (long long)r);
            return 5;
        }
    }
    std::fclose(fin);
    std::fclose(fout);
    return 0;
}
