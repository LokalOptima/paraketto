// bench_fp8_cutlass.cu — CUTLASS 3.x FP8 vs cublasLt FP8 on encoder GEMM shapes
//
// SM120 FP8 only supports TN layout: both A[M,K] and B[N,K] must be RowMajor
// (K-contiguous). This means weights must be stored transposed compared to the
// current cuBLAS convention. For the benchmark we store everything K-contiguous.
//
// Build: make bench_fp8_cutlass
// Usage: ./bench_fp8_cutlass

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// CUTLASS 3.x headers
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/float8.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// =========================================================================
// CUTLASS 3.x FP8 GEMM — SM120 TN layout (both operands K-contiguous)
// =========================================================================
//
// Computes: D[M,N] = A[M,K] @ B[N,K]^T
// where A is RowMajor (K contiguous) and B is RowMajor (K contiguous).
// This is the ONLY layout SM120 FP8 supports.

using ElementFp8          = cutlass::float_e4m3_t;
using ElementOut          = cutlass::half_t;
using ElementAccumulator  = float;
using ElementCompute      = float;

constexpr int AlignFp8 = 128 / cutlass::sizeof_bits<ElementFp8>::value;   // 16
constexpr int AlignOut = 128 / cutlass::sizeof_bits<ElementOut>::value;    // 8

template <int TileM, int TileN, int TileK>
struct Fp8Gemm {
    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape{}));
    using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementFp8, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, AlignFp8,
        ElementFp8, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, AlignFp8,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// =========================================================================
// Timing helper
// =========================================================================

struct Timer {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    Timer(cudaStream_t s) : stream(s) {
        cudaEventCreate(&start); cudaEventCreate(&stop);
    }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, stream); }
    float end() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// =========================================================================
// cublasLt FP8 GEMM wrapper
// =========================================================================
//
// TN layout: C[M,N] = A[M,K] @ B[N,K]^T
// A[M,K] RowMajor (K contiguous, ld=K) = A_col[K,M] ColumnMajor (ld=K... wait no)
//
// cuBLAS column-major trick:
//   A[M,K] row-major  ←→  A'[K,M] col-major with ld=K
//   B[N,K] row-major  ←→  B'[K,N] col-major with ld=K
//   C[M,N] row-major  ←→  C'[N,M] col-major with ld=N
//
// We want: C'[N,M] = op1 @ op2
// We need result shape [N,M], so cublas M=N, N=M.
//   op1 must be [N,K]: B'[K,N] with CUBLAS_OP_T → [N,K] ✓
//   op2 must be [K,M]: A'[K,M] with CUBLAS_OP_N → [K,M] ✓
//
// cuBLAS call: M_cb=N, N_cb=M, K=K, A_cb=B' opT ldA=K, B_cb=A' opN ldB=K, C_cb=C' ldC=N

static cublasHandle_t s_cublas;
static cublasLtHandle_t s_cublaslt;
static void* s_workspace;
static size_t s_workspace_size = 32 * 1024 * 1024;

// Persistent plan cache for cuBLAS (avoids re-creating descriptors each call)
struct CublasFp8Plan {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatmulAlgo_t algo;
    bool valid = false;
};
static CublasFp8Plan s_cublas_plan;

void cublaslt_fp8_init_plan(int M, int N, int K,
                             const float* a_scale_d, const float* b_scale_d) {
    auto& p = s_cublas_plan;
    if (p.valid) {
        cublasLtMatmulDescDestroy(p.desc);
        cublasLtMatrixLayoutDestroy(p.layoutA);
        cublasLtMatrixLayoutDestroy(p.layoutB);
        cublasLtMatrixLayoutDestroy(p.layoutC);
    }

    // TN: cuBLAS M=N, N=M
    cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale_d, sizeof(b_scale_d)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale_d, sizeof(a_scale_d)));

    // A_cb = B'[K,N] col-major, opT → [N,K]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.layoutA, CUDA_R_8F_E4M3, K, N, K));
    // B_cb = A'[K,M] col-major, opN → [K,M]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.layoutB, CUDA_R_8F_E4M3, K, M, K));
    // C_cb = C'[N,M] col-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.layoutC, CUDA_R_16F, N, M, N));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &s_workspace_size, sizeof(s_workspace_size)));
    cublasLtMatmulHeuristicResult_t result;
    int returned;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_cublaslt, p.desc, p.layoutA, p.layoutB,
                                                 p.layoutC, p.layoutC, pref, 1, &result, &returned));
    p.algo = result.algo;
    cublasLtMatmulPreferenceDestroy(pref);
    p.valid = true;
}

void cublaslt_fp8_run(cudaStream_t stream,
                       const uint8_t* A, const uint8_t* B, half* C) {
    float alpha = 1.0f, beta = 0.0f;
    auto& p = s_cublas_plan;
    // A_cb = B (weights), B_cb = A (activations)
    CUBLAS_CHECK(cublasLtMatmul(s_cublaslt, p.desc, &alpha,
                                B, p.layoutA, A, p.layoutB,
                                &beta, C, p.layoutC, C, p.layoutC,
                                &p.algo, s_workspace, s_workspace_size, stream));
}

// =========================================================================
// CUTLASS FP8 runner
// =========================================================================

template <typename GemmType>
struct CutlassFp8Runner {
    using Gemm = typename GemmType::Gemm;
    using ScaleConfig = typename GemmType::ScaleConfig;
    using LayoutSFA = typename GemmType::LayoutSFA;
    using LayoutSFB = typename GemmType::LayoutSFB;

    Gemm gemm_op;
    size_t workspace_size = 0;
    void* workspace = nullptr;
    bool initialized = false;

    // Pre-allocated scale buffers
    float* sfa_d = nullptr;
    float* sfb_d = nullptr;

    void init(cudaStream_t stream, int M, int N, int K) {
        auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

        int sfa_size = size(filter_zeros(layout_sfa));
        int sfb_size = size(filter_zeros(layout_sfb));

        CUDA_CHECK(cudaMalloc(&sfa_d, sfa_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&sfb_d, sfb_size * sizeof(float)));

        // Fill with 1.0 (per-tensor scaling with scale=1)
        std::vector<float> ones_a(sfa_size, 1.0f);
        std::vector<float> ones_b(sfb_size, 1.0f);
        CUDA_CHECK(cudaMemcpy(sfa_d, ones_a.data(), sfa_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sfb_d, ones_b.data(), sfb_size * sizeof(float), cudaMemcpyHostToDevice));

        initialized = true;
    }

    // Returns false if kernel can't implement this shape
    bool setup(cudaStream_t stream, int M, int N, int K,
               const ElementFp8* A, const ElementFp8* B,
               ElementOut* C) {

        auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

        auto stride_A = cutlass::make_cute_packed_stride(typename GemmType::StrideA{}, make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(typename GemmType::StrideB{}, make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(typename GemmType::StrideC{}, make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(typename GemmType::StrideD{}, make_shape(M, N, 1));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {A, stride_A,
             B, stride_B,
             sfa_d, layout_sfa,
             sfb_d, layout_sfb},
            {{1.0f, 0.0f},
             C, stride_C,
             C, stride_D}
        };

        auto status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS can_implement failed for %dx%dx%d: %d\n", M, N, K, (int)status);
            return false;
        }

        workspace_size = Gemm::get_workspace_size(args);
        if (workspace_size > 0 && !workspace) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }

        gemm_op.initialize(args, workspace, stream);
        return true;
    }

    void run(cudaStream_t stream) {
        gemm_op.run(stream);
    }

    ~CutlassFp8Runner() {
        if (sfa_d) cudaFree(sfa_d);
        if (sfb_d) cudaFree(sfb_d);
        if (workspace) cudaFree(workspace);
    }
};

// =========================================================================
// Benchmark one shape
// =========================================================================

// Instantiate kernel types to test
using Gemm_128x128x128 = Fp8Gemm<128, 128, 128>;
using Gemm_128x64x128  = Fp8Gemm<128, 64, 128>;
using Gemm_128x64x64   = Fp8Gemm<128, 64, 64>;

// Pingpong variant — uses KernelTmaWarpSpecializedBlockwisePingpongSm120
template <int TileM, int TileN, int TileK>
struct Fp8GemmPingpong {
    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape{}));
    using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementFp8, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, AlignFp8,
        ElementFp8, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, AlignFp8,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

using Gemm_PP_128x64x128 = Fp8GemmPingpong<128, 64, 128>;

// Non-blockwise FP8 GEMM — no scale factors, plain layouts
// Uses sm120_mma_builder.inl which accepts FP8 elements directly
template <int TileM, int TileN, int TileK>
struct Fp8GemmDense {
    using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        ElementOut, cutlass::layout::RowMajor, AlignOut,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    // Plain layouts (no scale factor tuple) → hits sm120_mma_builder.inl
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementFp8, cutlass::layout::RowMajor, AlignFp8,
        ElementFp8, cutlass::layout::ColumnMajor, AlignFp8,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

using Gemm_Dense_128x64x128 = Fp8GemmDense<128, 64, 128>;
using Gemm_Dense_128x64x64  = Fp8GemmDense<128, 64, 64>;

// Runner for dense (non-blockwise) FP8 kernels — no scale factors
template <typename GemmType>
struct CutlassFp8DenseRunner {
    using Gemm = typename GemmType::Gemm;

    Gemm gemm_op;
    size_t workspace_size = 0;
    void* workspace = nullptr;

    bool setup(cudaStream_t stream, int M, int N, int K,
               const ElementFp8* A, const ElementFp8* B,
               ElementOut* C) {

        auto stride_A = cutlass::make_cute_packed_stride(typename GemmType::StrideA{}, make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(typename GemmType::StrideB{}, make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(typename GemmType::StrideC{}, make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(typename GemmType::StrideD{}, make_shape(M, N, 1));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {A, stride_A,
             B, stride_B},
            {{1.0f, 0.0f},
             C, stride_C,
             C, stride_D}
        };

        auto status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS dense can_implement failed for %dx%dx%d: %d\n", M, N, K, (int)status);
            return false;
        }

        workspace_size = Gemm::get_workspace_size(args);
        if (workspace_size > 0 && !workspace) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }

        gemm_op.initialize(args, workspace, stream);
        return true;
    }

    void run(cudaStream_t stream) {
        gemm_op.run(stream);
    }

    ~CutlassFp8DenseRunner() {
        if (workspace) cudaFree(workspace);
    }
};

struct GemmShape {
    const char* name;
    int m, n, k;
};

template <typename GemmType>
void bench_shape(cudaStream_t stream, const GemmShape& s,
                 uint8_t* fp8_buf, half* fp16_buf,
                 float* scale_a_d, float* scale_b_d,
                 int warmup, int iters,
                 const char* tile_label) {

    // Data layout (TN = both K-contiguous):
    //   A[M,K] RowMajor at fp8_buf
    //   B[N,K] RowMajor at fp8_buf + M*K
    //   C[M,N] RowMajor at fp16_buf (cuBLAS) and fp16_buf + M*N*2 (CUTLASS)
    uint8_t* A_fp8 = fp8_buf;
    uint8_t* B_fp8 = fp8_buf + s.m * s.k;
    half* C_cublas  = fp16_buf;
    half* C_cutlass = fp16_buf + s.m * s.n;

    Timer t(stream);

    // --- cublasLt FP8 ---
    cublaslt_fp8_init_plan(s.m, s.n, s.k, scale_a_d, scale_b_d);

    for (int i = 0; i < warmup; i++)
        cublaslt_fp8_run(stream, A_fp8, B_fp8, C_cublas);
    cudaStreamSynchronize(stream);

    t.begin();
    for (int i = 0; i < iters; i++)
        cublaslt_fp8_run(stream, A_fp8, B_fp8, C_cublas);
    float cublas_us = t.end() / iters * 1000.0f;

    // --- CUTLASS FP8 ---
    CutlassFp8Runner<GemmType> runner;
    runner.init(stream, s.m, s.n, s.k);

    auto* cutlass_A = reinterpret_cast<const ElementFp8*>(A_fp8);
    auto* cutlass_B = reinterpret_cast<const ElementFp8*>(B_fp8);
    auto* cutlass_C = reinterpret_cast<ElementOut*>(C_cutlass);

    if (!runner.setup(stream, s.m, s.n, s.k, cutlass_A, cutlass_B, cutlass_C)) {
        printf("%-22s %4dx%-4dx%-4d %-12s  cublas=%7.2fus  cutlass=  SKIP\n",
               s.name, s.m, s.n, s.k, tile_label, cublas_us);
        return;
    }

    for (int i = 0; i < warmup; i++)
        runner.run(stream);
    cudaStreamSynchronize(stream);

    t.begin();
    for (int i = 0; i < iters; i++)
        runner.run(stream);
    float cutlass_us = t.end() / iters * 1000.0f;

    float ratio = cutlass_us / cublas_us;
    const char* verdict = ratio < 0.98f ? "CUTLASS" : ratio > 1.02f ? "cuBLAS" : "tie";
    printf("%-22s %4dx%-4dx%-4d %-12s  cublas=%7.2fus  cutlass=%7.2fus  ratio=%.3f  %s\n",
           s.name, s.m, s.n, s.k, tile_label,
           cublas_us, cutlass_us, ratio, verdict);
}

// =========================================================================
// Main
// =========================================================================

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUBLAS_CHECK(cublasCreate(&s_cublas));
    CUBLAS_CHECK(cublasSetStream(s_cublas, stream));
    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));
    CUDA_CHECK(cudaMalloc(&s_workspace, s_workspace_size));

    // Allocate buffers
    uint8_t* fp8_buf;
    half* fp16_buf;
    CUDA_CHECK(cudaMalloc(&fp8_buf, 256 * 1024 * 1024));
    CUDA_CHECK(cudaMalloc(&fp16_buf, 256 * 1024 * 1024));
    CUDA_CHECK(cudaMemset(fp8_buf, 0x3C, 256 * 1024 * 1024));
    CUDA_CHECK(cudaMemset(fp16_buf, 0, 256 * 1024 * 1024));

    // Per-tensor scales on device
    float scale_val = 1.0f;
    float *scale_a_d, *scale_b_d;
    CUDA_CHECK(cudaMalloc(&scale_a_d, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scale_b_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(scale_a_d, &scale_val, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scale_b_d, &scale_val, sizeof(float), cudaMemcpyHostToDevice));

    const int warmup = 50;
    const int iters = 200;

    // All shapes expressed as TN: D[M,N] = A[M,K] @ B[N,K]
    // For user-level NN (Y[m,n] = X[m,k] @ W[k,n]):
    //   M=m, K=k, N=n, A=X, B=W^T (weights transposed to [n,k] RowMajor)
    // For user-level NT (Y[m,n] = X[m,k] @ W[n,k]^T):
    //   M=m, K=k, N=n, A=X, B=W (weights already [n,k] RowMajor)

    for (int T : {128, 256, 512}) {
        printf("\n=== T=%d ===\n", T);
        int pos_len = 2 * T - 1;

        GemmShape shapes[] = {
            {"FF linear1",   T, 4096, 1024},
            {"FF linear2",   T, 1024, 4096},
            {"Fused QKV",    T, 3072, 1024},
            {"Attn out",     T, 1024, 1024},
            {"Conv pw1",     T, 2048, 1024},
            {"Conv pw2",     T, 1024, 1024},
            {"Enc proj",     T,  640, 1024},
        };

        for (auto& s : shapes) {
            bench_shape<Gemm_128x64x128> (stream, s, fp8_buf, fp16_buf, scale_a_d, scale_b_d, warmup, iters, "bw128x64");

            // Dense (no blockwise scaling) variants
            {
                uint8_t* A_fp8 = fp8_buf;
                uint8_t* B_fp8 = fp8_buf + s.m * s.k;
                half* C_cutlass = fp16_buf + s.m * s.n;

                auto run_dense = [&]<typename GT>(GT*, const char* label) {
                    CutlassFp8DenseRunner<GT> runner;
                    auto* cA = reinterpret_cast<const ElementFp8*>(A_fp8);
                    auto* cB = reinterpret_cast<const ElementFp8*>(B_fp8);
                    auto* cC = reinterpret_cast<ElementOut*>(C_cutlass);
                    if (!runner.setup(stream, s.m, s.n, s.k, cA, cB, cC)) {
                        printf("%-22s %4dx%-4dx%-4d %-12s  cublas=%7.2fus  cutlass=  SKIP\n",
                               s.name, s.m, s.n, s.k, label,
                               0.0f);  // cuBLAS already measured above
                        return;
                    }
                    for (int i = 0; i < warmup; i++) runner.run(stream);
                    cudaStreamSynchronize(stream);
                    Timer t2(stream);
                    t2.begin();
                    for (int i = 0; i < iters; i++) runner.run(stream);
                    float us = t2.end() / iters * 1000.0f;

                    // Re-measure cuBLAS for fair comparison
                    cublaslt_fp8_init_plan(s.m, s.n, s.k, scale_a_d, scale_b_d);
                    for (int i = 0; i < warmup; i++)
                        cublaslt_fp8_run(stream, A_fp8, B_fp8, (half*)fp16_buf);
                    cudaStreamSynchronize(stream);
                    Timer t3(stream);
                    t3.begin();
                    for (int i = 0; i < iters; i++)
                        cublaslt_fp8_run(stream, A_fp8, B_fp8, (half*)fp16_buf);
                    float cb_us = t3.end() / iters * 1000.0f;

                    float ratio = us / cb_us;
                    const char* v = ratio < 0.98f ? "CUTLASS" : ratio > 1.02f ? "cuBLAS" : "tie";
                    printf("%-22s %4dx%-4dx%-4d %-12s  cublas=%7.2fus  cutlass=%7.2fus  ratio=%.3f  %s\n",
                           s.name, s.m, s.n, s.k, label, cb_us, us, ratio, v);
                };

                run_dense((Gemm_Dense_128x64x128*)nullptr, "dense128x64");
                run_dense((Gemm_Dense_128x64x64*)nullptr,  "dense64x64");
            }
            printf("\n");
        }
    }

    cudaFree(fp8_buf);
    cudaFree(fp16_buf);
    cudaFree(scale_a_d);
    cudaFree(scale_b_d);
    cudaFree(s_workspace);
    cublasDestroy(s_cublas);
    cublasLtDestroy(s_cublaslt);
    cudaStreamDestroy(stream);

    return 0;
}
