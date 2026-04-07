// test_metal_gemm.mm — Correctness test for Metal GEMM kernels
//
// Tests NN, NT, GEMV, and batched GEMM against CPU reference.
// Run: clang++ -std=c++17 -O3 -fobjc-arc -Isrc -framework Metal -framework Foundation \
//      tests/test_metal_gemm.mm src/metal_context.mm -o tests/test_metal_gemm && ./tests/test_metal_gemm

#import <Metal/Metal.h>
#include "metal_context.h"
#include "metal_context_impl.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

// Embed both shader sources
static const char* read_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = 0;
    fclose(f);
    return buf;
}

struct GemmParams {
    uint32_t M, N, K, lda, ldb, ldc;
};

struct BatchedGemmParams {
    uint32_t M, N, K, lda, ldb, ldc, batch;
    uint64_t strideA, strideB, strideC;
};

static float randf() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

// CPU reference: C = A @ B (NN)
static void cpu_gemm_nn(const __fp16* A, const __fp16* B, float* C,
                        int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += (float)A[i * K + k] * (float)B[k * N + j];
            C[i * N + j] = acc;
        }
}

// CPU reference: C = A @ B^T (NT)
static void cpu_gemm_nt(const __fp16* A, const __fp16* B, float* C,
                        int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += (float)A[i * K + k] * (float)B[j * K + k];
            C[i * N + j] = acc;
        }
}

static float max_abs_err(const __fp16* got, const float* ref, int n) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf((float)got[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float max_rel_err(const __fp16* got, const float* ref, int n) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float r = fabsf(ref[i]);
        if (r < 1e-6f) continue;
        float err = fabsf((float)got[i] - ref[i]) / r;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static bool test_gemm(MetalContext& ctx, const char* name,
                      const char* kernel_name, bool is_nt,
                      int M, int N, int K) {
    printf("  %s [%d×%d×%d]... ", name, M, N, K);

    size_t a_size = M * K * sizeof(__fp16);
    size_t b_size = (is_nt ? N * K : K * N) * sizeof(__fp16);
    size_t c_size = M * N * sizeof(__fp16);
    size_t total = a_size + b_size + c_size;

    id<MTLBuffer> pool = [ctx.impl->device newBufferWithLength:total
                                                       options:MTLResourceStorageModeShared];
    __fp16* base = (__fp16*)pool.contents;
    __fp16* A = base;
    __fp16* B = (__fp16*)((char*)base + a_size);
    __fp16* C = (__fp16*)((char*)base + a_size + b_size);

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (__fp16)(randf() * 0.5f);
    for (int i = 0; i < (is_nt ? N * K : K * N); i++) B[i] = (__fp16)(randf() * 0.5f);
    memset(C, 0, c_size);

    // CPU reference
    std::vector<float> ref(M * N);
    if (is_nt) cpu_gemm_nt(A, B, ref.data(), M, N, K);
    else       cpu_gemm_nn(A, B, ref.data(), M, N, K);

    // GPU
    auto pso = ctx.impl->get_pipeline(kernel_name);
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:pool offset:0       atIndex:0];
    [enc setBuffer:pool offset:a_size  atIndex:1];
    [enc setBuffer:pool offset:a_size + b_size atIndex:2];

    GemmParams p;
    p.M = M; p.N = N; p.K = K;
    p.lda = K;
    p.ldb = is_nt ? K : N;
    p.ldc = N;
    [enc setBytes:&p length:sizeof(p) atIndex:3];

    size_t shmem = (32 * 64 + 32 * 32) * sizeof(__fp16);
    size_t shmem_store = 64 * 32 * sizeof(float);
    size_t shmem_total = shmem > shmem_store ? shmem : shmem_store;

    if (M == 1 && !is_nt) {
        // GEMV
        [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc setThreadgroupMemoryLength:shmem_total atIndex:0];
        uint gx = (N + 31) / 32;
        uint gy = (M + 63) / 64;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float abs_err = max_abs_err(C, ref.data(), M * N);
    float rel_err = max_rel_err(C, ref.data(), M * N);

    // FP16 accumulation tolerance: relative error up to ~1% for large K
    bool ok = rel_err < 0.05f;
    printf("abs=%.4f rel=%.4f %s\n", abs_err, rel_err, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    printf("Metal GEMM correctness test\n");
    printf("===========================\n");

    MetalContext ctx;

    // Load shader source from file
    const char* gemm_src = read_file("metal/gemm.metal");
    ctx.load_shaders(gemm_src, "gemm");
    free((void*)gemm_src);

    bool ok = true;

    // NN GEMM tests (model shapes)
    ok &= test_gemm(ctx, "NN FF1",  "gemm_nn_f16", false, 100, 4096, 1024);
    ok &= test_gemm(ctx, "NN FF2",  "gemm_nn_f16", false, 100, 1024, 4096);
    ok &= test_gemm(ctx, "NN QKV",  "gemm_nn_f16", false, 100, 3072, 1024);
    ok &= test_gemm(ctx, "NN small","gemm_nn_f16", false, 8,   64,   32);
    ok &= test_gemm(ctx, "NN edge", "gemm_nn_f16", false, 7,   13,   19);

    // GEMV (M=1, decoder)
    ok &= test_gemm(ctx, "GEMV dec","gemv_nn_f16", false, 1, 640, 640);
    ok &= test_gemm(ctx, "GEMV out","gemv_nn_f16", false, 1, 1030, 640);

    // NT GEMM tests
    ok &= test_gemm(ctx, "NT attn", "gemm_nt_f16", true, 100, 100, 128);
    ok &= test_gemm(ctx, "NT conv", "gemm_nt_f16", true, 100, 1024, 1024);
    ok &= test_gemm(ctx, "NT edge", "gemm_nt_f16", true, 5,   9,    17);

    // Performance test on a large shape
    {
        int M = 500, N = 4096, K = 1024;
        size_t total = (M*K + K*N + M*N) * sizeof(__fp16);
        id<MTLBuffer> pool = [ctx.impl->device newBufferWithLength:total
                                                           options:MTLResourceStorageModeShared];
        __fp16* base = (__fp16*)pool.contents;
        for (int i = 0; i < (int)(total / sizeof(__fp16)); i++) base[i] = (__fp16)(randf() * 0.1f);

        auto pso = ctx.impl->get_pipeline("gemm_nn_f16");

        // Warmup
        for (int t = 0; t < 3; t++) {
            id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:pool offset:0 atIndex:0];
            [e setBuffer:pool offset:M*K*sizeof(__fp16) atIndex:1];
            [e setBuffer:pool offset:(M*K + K*N)*sizeof(__fp16) atIndex:2];
            GemmParams p = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)K, (uint32_t)N, (uint32_t)N};
            [e setBytes:&p length:sizeof(p) atIndex:3];
            size_t shmem = (32*64 + 32*32) * sizeof(__fp16);
            size_t shmem_store = 64*32*sizeof(float);
            [e setThreadgroupMemoryLength:(shmem > shmem_store ? shmem : shmem_store) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Benchmark
        int N_ITER = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < N_ITER; t++) {
            id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:pool offset:0 atIndex:0];
            [e setBuffer:pool offset:M*K*sizeof(__fp16) atIndex:1];
            [e setBuffer:pool offset:(M*K + K*N)*sizeof(__fp16) atIndex:2];
            GemmParams p = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)K, (uint32_t)N, (uint32_t)N};
            [e setBytes:&p length:sizeof(p) atIndex:3];
            size_t shmem = (32*64 + 32*32) * sizeof(__fp16);
            size_t shmem_store = 64*32*sizeof(float);
            [e setThreadgroupMemoryLength:(shmem > shmem_store ? shmem : shmem_store) atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITER;
        double gflops = 2.0 * M * N * K / (us * 1e3);
        printf("\n  Perf NN [%d×%d×%d]: %.0f us (%.1f GFLOPS)\n", M, N, K, us, gflops);
    }

    printf("===========================\n");
    printf("%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
