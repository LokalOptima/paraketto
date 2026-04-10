// test_fp8_2x.cu — CUTLASS 2.x FP8 vs cublasLt FP8 on all conformer shapes
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/float8.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// CUTLASS 2.x FP8 kernel — SM89 MMA (m16n8k32), works on SM120
template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using GemmFP8 = cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    Epilogue<8>, Swizzle, 6, 16, 16>;

// cublasLt FP8
static cublasHandle_t s_cublas;
static cublasLtHandle_t s_cublaslt;
static void* s_workspace;
static size_t s_ws_size = 32 * 1024 * 1024;

struct CublasLtPlan {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t lA, lB, lC;
    cublasLtMatmulAlgo_t algo;
};

CublasLtPlan make_plan(int M, int N, int K, const float* sa, const float* sb) {
    CublasLtPlan p;
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.lA, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.lB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.lC, CUDA_R_16F, N, M, N));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &s_ws_size, sizeof(s_ws_size)));
    cublasLtMatmulHeuristicResult_t res; int ret;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_cublaslt, p.desc, p.lA, p.lB, p.lC, p.lC, pref, 1, &res, &ret));
    p.algo = res.algo;
    cublasLtMatmulPreferenceDestroy(pref);
    return p;
}

void run_cublas(cudaStream_t s, CublasLtPlan& p, const uint8_t* A, const uint8_t* B, half* C) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(s_cublaslt, p.desc, &alpha, B, p.lA, A, p.lB,
                                        &beta, C, p.lC, C, p.lC, &p.algo, s_workspace, s_ws_size, s);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasLtMatmul failed: %d\n", (int)st); exit(1); }
}

struct Shape { const char* name; int m, n, k; int count; };

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&s_cublas));
    CUBLAS_CHECK(cublasSetStream(s_cublas, stream));
    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));
    CUDA_CHECK(cudaMalloc(&s_workspace, s_ws_size));

    uint8_t* fp8_buf; half* fp16_buf;
    CUDA_CHECK(cudaMalloc(&fp8_buf, 256*1024*1024));
    CUDA_CHECK(cudaMalloc(&fp16_buf, 256*1024*1024));
    CUDA_CHECK(cudaMemset(fp8_buf, 0x3C, 256*1024*1024));

    float sv = 1.0f, *sa, *sb;
    CUDA_CHECK(cudaMalloc(&sa, 4)); CUDA_CHECK(cudaMalloc(&sb, 4));
    CUDA_CHECK(cudaMemcpy(sa, &sv, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sb, &sv, 4, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int warmup = 50, iters = 200;

    for (int T : {63, 131, 340}) {
        int pos = 2*T - 1;

        // Per conformer block: 9 GEMMs. 24 blocks + 1 enc_proj
        Shape shapes[] = {
            {"FF1 linear1",  T, 4096, 1024, 24},
            {"FF1 linear2",  T, 1024, 4096, 24},
            {"Fused QKV",    T, 3072, 1024, 24},
            {"Pos proj",     pos, 1024, 1024, 24},
            {"Attn out",     T, 1024, 1024, 24},
            {"Conv pw1",     T, 2048, 1024, 24},
            {"Conv pw2",     T, 1024, 1024, 24},
            {"FF2 linear1",  T, 4096, 1024, 24},
            {"FF2 linear2",  T, 1024, 4096, 24},
            {"Enc proj",     T,  640, 1024,  1},
        };

        printf("\n=== T=%d ===\n", T);
        printf("%-15s %4s %-4s %-4s  x    %9s  %9s  ratio\n",
               "shape", "M", "N", "K", "cuBLAS", "CUTLASS");

        float total_cublas = 0, total_cutlass = 0;

        for (auto& s : shapes) {
            // cuBLAS
            auto plan = make_plan(s.m, s.n, s.k, sa, sb);
            for (int i = 0; i < warmup; i++) run_cublas(stream, plan, fp8_buf, fp8_buf + s.m*s.k, fp16_buf);
            cudaStreamSynchronize(stream);
            cudaEventRecord(t0, stream);
            for (int i = 0; i < iters; i++) run_cublas(stream, plan, fp8_buf, fp8_buf + s.m*s.k, fp16_buf);
            cudaEventRecord(t1, stream); cudaEventSynchronize(t1);
            float cb_ms; cudaEventElapsedTime(&cb_ms, t0, t1);
            float cb_us = cb_ms / iters * 1000;

            // CUTLASS 2.x
            GemmFP8::Arguments args({s.m, s.n, s.k},
                {reinterpret_cast<cutlass::float_e4m3_t*>(fp8_buf), s.k},
                {reinterpret_cast<cutlass::float_e4m3_t*>(fp8_buf + s.m*s.k), s.k},
                {reinterpret_cast<cutlass::half_t*>(fp16_buf + s.m*s.n), s.n},
                {reinterpret_cast<cutlass::half_t*>(fp16_buf + s.m*s.n), s.n},
                {1.0f, 0.0f});
            GemmFP8 gemm;
            gemm.initialize(args, nullptr, stream);
            for (int i = 0; i < warmup; i++) gemm(stream);
            cudaStreamSynchronize(stream);
            cudaEventRecord(t0, stream);
            for (int i = 0; i < iters; i++) gemm(stream);
            cudaEventRecord(t1, stream); cudaEventSynchronize(t1);
            float cl_ms; cudaEventElapsedTime(&cl_ms, t0, t1);
            float cl_us = cl_ms / iters * 1000;

            float ratio = cl_us / cb_us;
            printf("%-15s %4dx%-4dx%-4d x%-3d %7.2fus  %7.2fus  %.2f\n",
                   s.name, s.m, s.n, s.k, s.count, cb_us, cl_us, ratio);

            total_cublas  += cb_us * s.count;
            total_cutlass += cl_us * s.count;

            cublasLtMatmulDescDestroy(plan.desc);
            cublasLtMatrixLayoutDestroy(plan.lA);
            cublasLtMatrixLayoutDestroy(plan.lB);
            cublasLtMatrixLayoutDestroy(plan.lC);
        }

        printf("%-15s %38s %7.1fus  %7.1fus  %.2f\n",
               "TOTAL FP8", "", total_cublas, total_cutlass, total_cutlass / total_cublas);
    }

    cudaFree(fp8_buf); cudaFree(fp16_buf);
    cudaFree(sa); cudaFree(sb); cudaFree(s_workspace);
    return 0;
}
