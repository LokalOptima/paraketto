// bench_fp8_tiles.cu — Sweep CUTLASS 2.x FP8 tile configs vs cuBLAS
//
// Build: make bench_fp8_tiles
// Usage: ./bench_fp8_tiles

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/float8.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// =========================================================================
// FP8 E4M3 kernel types — SM89 MMA (m16n8k32), TN layout
// =========================================================================

using E8 = cutlass::float_e4m3_t;
using EOut = cutlass::half_t;
using RA = cutlass::layout::RowMajor;
using CB = cutlass::layout::ColumnMajor;
using RO = cutlass::layout::RowMajor;
using Sw = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

template<int A> using Epi = cutlass::epilogue::thread::LinearCombination<EOut, A, float, float>;

// Tile configs: ThreadblockShape, WarpShape, stages, alignment
// MMA instruction is always 16x8x32 for FP8

// 64x64 tiles
using G_64x64_64_s6 = cutlass::gemm::device::Gemm<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64,64,64>, cutlass::gemm::GemmShape<32,32,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>, Sw, 6, 16, 16>;

using G_64x64_128_s3 = cutlass::gemm::device::Gemm<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64,64,128>, cutlass::gemm::GemmShape<32,32,128>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>, Sw, 3, 16, 16>;

// 128x64 tiles
using G_128x64_64_s6 = cutlass::gemm::device::Gemm<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128,64,64>, cutlass::gemm::GemmShape<64,32,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>, Sw, 6, 16, 16>;

// 64x128 tile
using G_64x128_64_s6 = cutlass::gemm::device::Gemm<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64,128,64>, cutlass::gemm::GemmShape<32,64,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>, Sw, 6, 16, 16>;

// 128x128 tile
using G_128x128_64_s3 = cutlass::gemm::device::Gemm<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128,128,64>, cutlass::gemm::GemmShape<64,64,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>, Sw, 3, 16, 16>;

// Split-K variants
using SK_64x64_64_s6 = cutlass::gemm::device::GemmSplitKParallel<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64,64,64>, cutlass::gemm::GemmShape<32,32,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>>;

using SK_128x64_64_s6 = cutlass::gemm::device::GemmSplitKParallel<
    E8, RA, E8, CB, EOut, RO, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128,64,64>, cutlass::gemm::GemmShape<64,32,64>,
    cutlass::gemm::GemmShape<16,8,32>, Epi<8>>;

// =========================================================================
// Runner helpers
// =========================================================================

template<typename GemmOp>
float bench_cutlass(int M, int N, int K, uint8_t* buf, half* out,
                    void* workspace, size_t ws_size,
                    cudaStream_t stream, cudaEvent_t t0, cudaEvent_t t1,
                    int warmup, int iters) {
    using EA = cutlass::float_e4m3_t;
    auto* A = reinterpret_cast<const EA*>(buf);
    auto* B = reinterpret_cast<const EA*>(buf + M * K);
    auto* C = reinterpret_cast<EOut*>(out);

    typename GemmOp::Arguments args({M, N, K},
        {A, K}, {B, K}, {C, N}, {C, N}, {1.0f, 0.0f});

    GemmOp op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1.0f;

    size_t ws = GemmOp::get_workspace_size(args);
    void* w = (ws > 0 && ws <= ws_size) ? workspace : nullptr;
    op.initialize(args, w, stream);

    for (int i = 0; i < warmup; i++) op(stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++) op(stream);
    cudaEventRecord(t1, stream); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return ms / iters * 1000.0f;
}

template<typename SKOp>
float bench_splitk(int M, int N, int K, int sk,
                   uint8_t* buf, half* out,
                   void* workspace, size_t ws_size,
                   cudaStream_t stream, cudaEvent_t t0, cudaEvent_t t1,
                   int warmup, int iters) {
    using EA = cutlass::float_e4m3_t;
    auto* A = reinterpret_cast<const EA*>(buf);
    auto* B = reinterpret_cast<const EA*>(buf + M * K);
    auto* C = reinterpret_cast<EOut*>(out);

    typename SKOp::Arguments args({M, N, K},
        {A, K}, {B, K}, {C, N}, {C, N}, {1.0f, 0.0f}, sk);

    SKOp op;
    size_t ws = SKOp::get_workspace_size(args);
    void* w = (ws > 0 && ws <= ws_size) ? workspace : nullptr;
    auto st = op.initialize(args, w);
    if (st != cutlass::Status::kSuccess) return -1.0f;

    for (int i = 0; i < warmup; i++) op.run(stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++) op.run(stream);
    cudaEventRecord(t1, stream); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return ms / iters * 1000.0f;
}

// cuBLAS reference
static cublasLtHandle_t s_lt;
static void* s_ws;
static size_t s_ws_size = 32*1024*1024;

float bench_cublas(int M, int N, int K,
                   uint8_t* buf, half* out, float* sa, float* sb,
                   cudaStream_t stream, cudaEvent_t t0, cudaEvent_t t1,
                   int warmup, int iters) {
    cublasLtMatmulDesc_t desc;
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa));

    cublasLtMatrixLayout_t lA, lB, lC;
    cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &s_ws_size, sizeof(s_ws_size));
    cublasLtMatmulHeuristicResult_t res; int ret;
    cublasLtMatmulAlgoGetHeuristic(s_lt, desc, lA, lB, lC, lC, pref, 1, &res, &ret);
    cublasLtMatmulPreferenceDestroy(pref);

    uint8_t* A = buf;
    uint8_t* B = buf + M * K;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup; i++)
        cublasLtMatmul(s_lt, desc, &alpha, B, lA, A, lB, &beta, out, lC, out, lC, &res.algo, s_ws, s_ws_size, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        cublasLtMatmul(s_lt, desc, &alpha, B, lA, A, lB, &beta, out, lC, out, lC, &res.algo, s_ws, s_ws_size, stream);
    cudaEventRecord(t1, stream); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(lA);
    cublasLtMatrixLayoutDestroy(lB);
    cublasLtMatrixLayoutDestroy(lC);
    return ms / iters * 1000.0f;
}

// =========================================================================
// Main
// =========================================================================

struct Shape { const char* name; int m, n, k; };

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasLtCreate(&s_lt);
    CUDA_CHECK(cudaMalloc(&s_ws, s_ws_size));

    uint8_t* fp8_buf; half* fp16_buf;
    CUDA_CHECK(cudaMalloc(&fp8_buf, 256*1024*1024));
    CUDA_CHECK(cudaMalloc(&fp16_buf, 256*1024*1024));
    CUDA_CHECK(cudaMemset(fp8_buf, 0x3C, 256*1024*1024));

    void* cutlass_ws;
    size_t cutlass_ws_size = 64*1024*1024;
    CUDA_CHECK(cudaMalloc(&cutlass_ws, cutlass_ws_size));

    float sv = 1.0f, *sa, *sb;
    CUDA_CHECK(cudaMalloc(&sa, 4)); CUDA_CHECK(cudaMalloc(&sb, 4));
    CUDA_CHECK(cudaMemcpy(sa, &sv, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sb, &sv, 4, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int warmup = 50, iters = 200;

    for (int T : {63, 131, 340}) {
        int pos = 2*T-1;
        printf("\n=== T=%d ===\n", T);

        Shape shapes[] = {
            {"FF linear1",  T, 4096, 1024},
            {"FF linear2",  T, 1024, 4096},
            {"Fused QKV",   T, 3072, 1024},
            {"Pos proj",    pos, 1024, 1024},
            {"Attn out",    T, 1024, 1024},
            {"Conv pw1",    T, 2048, 1024},
            {"Conv pw2",    T, 1024, 1024},
            {"Enc proj",    T,  640, 1024},
        };

        for (auto& s : shapes) {
            float cb = bench_cublas(s.m, s.n, s.k, fp8_buf, fp16_buf, sa, sb, stream, t0, t1, warmup, iters);

            struct { const char* name; float us; } results[20];
            int n = 0;

            #define TRY(TYPE, LABEL) { \
                float us = bench_cutlass<TYPE>(s.m, s.n, s.k, fp8_buf, fp16_buf, cutlass_ws, cutlass_ws_size, stream, t0, t1, warmup, iters); \
                if (us > 0) results[n++] = {LABEL, us}; \
            }
            #define TRY_SK(TYPE, SK, LABEL) { \
                float us = bench_splitk<TYPE>(s.m, s.n, s.k, SK, fp8_buf, fp16_buf, cutlass_ws, cutlass_ws_size, stream, t0, t1, warmup, iters); \
                if (us > 0) results[n++] = {LABEL, us}; \
            }

            TRY(G_64x64_64_s6, "64x64_64s6");
            TRY(G_64x64_128_s3, "64x64_128s3");
            TRY(G_128x64_64_s6, "128x64_64s6");
            TRY(G_64x128_64_s6, "64x128_64s6");
            TRY(G_128x128_64_s3, "128x128_64s3");
            TRY_SK(SK_64x64_64_s6, 2, "sk2_64x64");
            TRY_SK(SK_64x64_64_s6, 4, "sk4_64x64");
            TRY_SK(SK_128x64_64_s6, 2, "sk2_128x64");
            TRY_SK(SK_128x64_64_s6, 4, "sk4_128x64");

            // Find best
            int best = 0;
            for (int i = 1; i < n; i++)
                if (results[i].us < results[best].us) best = i;

            float ratio = results[best].us / cb;
            const char* v = ratio < 0.98f ? " <<<" : ratio > 1.05f ? "" : " ~";
            printf("%-12s %4dx%-4dx%-4d  cuBLAS=%5.1f  best=%-14s %5.1f  ratio=%.2f%s\n",
                   s.name, s.m, s.n, s.k, cb, results[best].name, results[best].us, ratio, v);
        }
    }

    cudaFree(fp8_buf); cudaFree(fp16_buf); cudaFree(cutlass_ws);
    cudaFree(sa); cudaFree(sb); cudaFree(s_ws);
    return 0;
}
