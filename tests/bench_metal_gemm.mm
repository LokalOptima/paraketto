// bench_metal_gemm.mm — Measure Metal GEMM throughput at conformer shapes
//
// Reports TFLOPS for each shape used in the encoder forward pass.
// M1 Max peak: ~10.4 TFLOPS FP16.

#import <Metal/Metal.h>
#include "metal_context.h"
#include "metal_context_impl.h"
#include "metal_gemm.h"
#include "common_metal.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

struct GemmShape {
    const char* name;
    int M, N, K;
    bool is_nt;  // NT transpose
};

int main() {
    MetalContext ctx;

    // Load shaders
    auto read_file = [](const char* path) -> std::string {
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        std::string s(len, '\0'); fread(&s[0], 1, len, f); fclose(f);
        return s;
    };
    std::string src = read_file("metal/kernels.metal");
    src += "\n";
    src += read_file("metal/gemm.metal");
    ctx.load_shaders(src.c_str(), "bench");
    metal_gemm_init(ctx);

    // Shapes from conformer forward pass (T≈131)
    std::vector<GemmShape> shapes = {
        {"FF1-expand  ", 131, 4096, 1024, false},
        {"FF1-contract", 131, 1024, 4096, false},
        {"QKV-proj    ", 131, 3072, 1024, false},
        {"Pos-proj    ", 261, 1024, 1024, false},
        {"Out-proj    ", 131, 1024, 1024, false},
        {"Conv-pw1 NT ", 131, 2048, 1024, true},
        {"Conv-pw2 NT ", 131, 1024, 1024, true},
        {"FF2-expand  ", 131, 4096, 1024, false},
        {"FF2-contract", 131, 1024, 4096, false},
        {"Sub-proj    ", 131, 1024, 4096, false},
        {"Enc-proj    ", 131,  640, 1024, false},
    };

    // Allocate buffer large enough for all shapes
    size_t max_elems = 0;
    for (auto& s : shapes) {
        size_t need = (size_t)s.M * s.K + (size_t)s.K * s.N + (size_t)s.M * s.N;
        if (need > max_elems) max_elems = need;
    }
    size_t pool_bytes = max_elems * sizeof(__fp16) + 4096;
    void* pool = ctx.alloc_shared(pool_bytes);

    // Zero-fill (content doesn't matter for timing)
    memset(ctx.buffer_contents(pool), 0, pool_bytes);

    int warmup = 5;
    int iters = 50;

    printf("%-14s  %5s %5s %5s  %8s  %7s  %6s\n",
           "Shape", "M", "N", "K", "GFLOPs", "ms", "TFLOPS");
    printf("─────────────────────────────────────────────────────────────\n");

    for (auto& s : shapes) {
        size_t X_off = 0;
        size_t W_off = (size_t)s.M * s.K * sizeof(__fp16);
        W_off = (W_off + 255) & ~(size_t)255;
        size_t Y_off = W_off + (size_t)s.K * s.N * sizeof(__fp16);
        Y_off = (Y_off + 255) & ~(size_t)255;

        double gflops = 2.0 * s.M * s.N * s.K / 1e9;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            void* E = (__bridge void*)enc;
            if (s.is_nt)
                metal_gemm_nt(ctx, E, pool, X_off, s.M, s.K, W_off, s.N, Y_off);
            else
                metal_gemm_nn(ctx, E, pool, X_off, s.M, s.K, W_off, s.N, Y_off);
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        }

        // Timed iterations
        double best_ms = 1e9;
        for (int trial = 0; trial < 3; trial++) {
            id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            void* E = (__bridge void*)enc;
            for (int i = 0; i < iters; i++) {
                if (s.is_nt)
                    metal_gemm_nt(ctx, E, pool, X_off, s.M, s.K, W_off, s.N, Y_off);
                else
                    metal_gemm_nn(ctx, E, pool, X_off, s.M, s.K, W_off, s.N, Y_off);
            }
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            double ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0 / iters;
            if (ms < best_ms) best_ms = ms;
        }

        double tflops = gflops / best_ms;
        printf("%-14s  %5d %5d %5d  %8.3f  %7.3f  %6.2f\n",
               s.name, s.M, s.N, s.K, gflops, best_ms, tflops);
    }

    ctx.free_buffer(pool);
    metal_gemm_free();
    return 0;
}
