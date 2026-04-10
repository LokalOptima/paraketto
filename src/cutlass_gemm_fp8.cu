// cutlass_gemm_fp8.cu — CUTLASS FP8 GEMM kernels replacing cublasLt
//
// Instantiates SM89 FP8 TensorOp templates (m16n8k32) with per-shape tile
// configs benchmarked against cublasLt FP8. All GEMMs use TN layout:
//   D[m,n] = A[m,k] (RowMajor) @ B[k,n] (ColumnMajor)
// where A is the FP8-quantized activation and B is the FP8 weight stored
// as [n,k] RowMajor (reinterpreted as [k,n] ColumnMajor — same physical
// layout, both K-contiguous).
//
// Alpha (dequantization scale product) is read from a device pointer via
// the CUTLASS LinearCombination epilogue's alpha_ptr mechanism, avoiding
// any host-device sync for scale values.
//
// Tile configs from bench_fp8_tiles.cu sweep (RTX 5070 Ti SM120):
//   FF linear2 (K=4096): split-K sk4/sk2 with 64x64 → 0.72-1.04x cuBLAS
//   FF linear1/QKV (N≥2048): 64x64x128 s3 or 64x128x64 s6 → 0.94-1.19x
//   Attn out/Conv pw2 (N=1024): 64x64x64 s6 or 64x128x64 s6 → 1.07-1.09x
//   Enc proj (N=640): 64x64x128 s3 → 0.98x
//
// Build: nvcc -std=c++17 -arch=sm_120 -O3 -I cutlass/include \
//        --expt-relaxed-constexpr -c cutlass_gemm_fp8.cu

#include "cutlass_gemm_fp8.h"
#include "common.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/float8.h"

namespace paraketto {

// =========================================================================
// CUTLASS FP8 kernel type aliases — SM89 TensorOp, TN layout
// =========================================================================
//
// All types use:
//   A = float_e4m3_t, RowMajor   (activation, K-contiguous)
//   B = float_e4m3_t, ColumnMajor (weight, K-contiguous)
//   C = half_t, RowMajor          (output FP16)
//   Accumulator = float            (FP32 accumulation)
//   MMA = m16n8k32                 (SM89 FP8 tensor op)
//   Alignment = 16                 (128-bit loads for FP8)

using E8  = cutlass::float_e4m3_t;
using EOut = cutlass::half_t;

template<int Align>
using FP8Epi = cutlass::epilogue::thread::LinearCombination<EOut, Align, float, float>;
using FP8Sw  = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// 64x64x64, 6 stages — default, good for medium shapes (N=1024, K=1024)
using FP8_64x64_64_s6 = cutlass::gemm::device::Gemm<
    E8, cutlass::layout::RowMajor,
    E8, cutlass::layout::ColumnMajor,
    EOut, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    FP8Epi<8>, FP8Sw, 6, 16, 16>;

// 64x64x128, 3 stages — deep K-tile, good for large-N (FF linear1 N=4096)
using FP8_64x64_128_s3 = cutlass::gemm::device::Gemm<
    E8, cutlass::layout::RowMajor,
    E8, cutlass::layout::ColumnMajor,
    EOut, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    FP8Epi<8>, FP8Sw, 3, 16, 16>;

// 64x128x64, 6 stages — wide N-tile, good for medium N=1024 at larger M
using FP8_64x128_64_s6 = cutlass::gemm::device::Gemm<
    E8, cutlass::layout::RowMajor,
    E8, cutlass::layout::ColumnMajor,
    EOut, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    FP8Epi<8>, FP8Sw, 6, 16, 16>;

// Split-K 64x64x64 — for large-K shapes (FF linear2, K=4096)
using FP8_SK_64x64_64 = cutlass::gemm::device::GemmSplitKParallel<
    E8, cutlass::layout::RowMajor,
    E8, cutlass::layout::ColumnMajor,
    EOut, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    FP8Epi<8>>;

// =========================================================================
// Workspace (for split-K reduction)
// =========================================================================

static void* s_fp8_workspace = nullptr;
static size_t s_fp8_workspace_size = 0;

void cutlass_fp8_gemm_init(cudaStream_t) {
    // Split-K workspace: 64x64 tile, sk=4, T=800:
    //   1024*800*sizeof(float)*4 ≈ 13MB. 32MB is safe.
    s_fp8_workspace_size = 32 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&s_fp8_workspace, s_fp8_workspace_size));
}

void cutlass_fp8_gemm_free() {
    if (s_fp8_workspace) { cudaFree(s_fp8_workspace); s_fp8_workspace = nullptr; }
}

// =========================================================================
// Alpha computation kernels
// =========================================================================

__global__ void compute_alpha_kernel(float* out, const float* a, const float* b) {
    *out = *a * *b;
}

void fp8_compute_alpha(float* out, const float* a, const float* b, cudaStream_t stream) {
    compute_alpha_kernel<<<1, 1, 0, stream>>>(out, a, b);
}

__global__ void compute_all_alphas_kernel(float* alphas, const float* w_scales,
                                          const float* act_scales, int n_blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_blocks * 9 + 2;
    if (i >= total) return;

    int w_idx;
    if (i < n_blocks * 9) {
        // Per-block mapping: act site offset → weight scale offset
        // 0(ff1_w1)→1, 1(ff1_w2)→2, 2(qkv)→0, 3(pos)→5, 4(out)→6,
        // 5(pw1)→7, 6(pw2)→8, 7(ff2_w1)→3, 8(ff2_w2)→4
        const int map[9] = {1, 2, 0, 5, 6, 7, 8, 3, 4};
        int blk = i / 9, off = i % 9;
        w_idx = blk * 9 + map[off];
    } else {
        // Global sites (sub_out, enc_proj): w_idx == act_idx
        w_idx = i;
    }
    alphas[i] = w_scales[w_idx] * act_scales[i];
}

void fp8_compute_all_alphas(float* alphas, const float* w_scales,
                            const float* act_scales, int n_blocks,
                            cudaStream_t stream) {
    compute_all_alphas_kernel<<<1, 256, 0, stream>>>(alphas, w_scales, act_scales, n_blocks);
}

// =========================================================================
// Generic CUTLASS FP8 GEMM runner
// =========================================================================

template<typename GemmOp>
static void run_fp8_gemm(int M, int N, int K,
                         const uint8_t* A, int ldA,
                         const uint8_t* B, int ldB,
                         half* C, int ldC,
                         const float* alpha_ptr,
                         cudaStream_t stream) {
    using EA = cutlass::float_e4m3_t;
    using EC = cutlass::half_t;

    typename GemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const EA*>(A), ldA},
        {reinterpret_cast<const EA*>(B), ldB},
        {reinterpret_cast<EC*>(C), ldC},
        {reinterpret_cast<EC*>(C), ldC},
        {alpha_ptr}  // device pointer for alpha, beta defaults to 0
    );

    GemmOp op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 can_implement failed for %dx%dx%d: %d\n",
                M, N, K, (int)status);
        exit(1);
    }

    size_t ws = GemmOp::get_workspace_size(args);
    void* w = (ws > 0 && ws <= s_fp8_workspace_size) ? s_fp8_workspace : nullptr;

    status = op.initialize(args, w, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 initialize failed for %dx%dx%d: %d\n",
                M, N, K, (int)status);
        exit(1);
    }

    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 run failed for %dx%dx%d: %d\n",
                M, N, K, (int)status);
        exit(1);
    }
}

// =========================================================================
// Generic CUTLASS FP8 split-K parallel runner
// =========================================================================

template<typename SKOp>
static void run_fp8_splitk(int M, int N, int K, int sk,
                           const uint8_t* A, int ldA,
                           const uint8_t* B, int ldB,
                           half* C, int ldC,
                           const float* alpha_ptr,
                           cudaStream_t stream) {
    using EA = cutlass::float_e4m3_t;
    using EC = cutlass::half_t;

    typename SKOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const EA*>(A), ldA},
        {reinterpret_cast<const EA*>(B), ldB},
        {reinterpret_cast<EC*>(C), ldC},
        {reinterpret_cast<EC*>(C), ldC},
        {alpha_ptr},
        sk
    );

    SKOp op;
    size_t ws = SKOp::get_workspace_size(args);
    void* w = (ws > 0 && ws <= s_fp8_workspace_size) ? s_fp8_workspace : nullptr;

    auto status = op.initialize(args, w);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 split-K init failed for %dx%dx%d sk=%d: %d\n",
                M, N, K, sk, (int)status);
        exit(1);
    }

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 split-K run failed for %dx%dx%d sk=%d: %d\n",
                M, N, K, sk, (int)status);
        exit(1);
    }
}

// =========================================================================
// Dispatch: select best tile config based on shape
// =========================================================================
//
// Shape-based config selection (from bench_fp8_tiles.cu sweep):
//
//   K≥4096 (FF linear2, sub_out): split-K dominates
//     M≤128: sk4_64x64 (0.72x cuBLAS at T=63, 0.92x at T=131)
//     M>128:  sk2_64x64 (1.04x at T=340)
//
//   N≥2048, K=1024 (FF linear1, QKV, Conv pw1):
//     M≤64:  64x64_128s3 (0.94x FF linear1 at T=63)
//     M>64:  64x128_64s6 (0.97x Conv pw1 at T=131)
//
//   N=1024, K=1024 (Pos proj, Attn out, Conv pw2):
//     M≤128: 64x64_64s6
//     M>128: 64x128_64s6 (1.07x at T=340)
//
//   N<1024 (Enc proj N=640):
//     64x64_128s3 (0.98x at T=340)
//
//   Default: 64x64_64s6

void cutlass_fp8_gemm(cudaStream_t stream,
                      const uint8_t* X_fp8, int m, int k,
                      const uint8_t* W_fp8, int n,
                      const float* alpha_ptr, half* Y) {
    // CUTLASS TN: D[m,n] = A[m,k] RowMajor @ B[k,n] ColumnMajor
    // A = X_fp8 activation, B = W_fp8 weight ([n,k] RM = [k,n] CM)
    int ldA = k, ldB = k, ldC = n;

    if (k >= 4096) {
        // FF linear2 / sub_out: large K, smaller N — split-K
        if (m <= 128)
            run_fp8_splitk<FP8_SK_64x64_64>(m, n, k, 4, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
        else
            run_fp8_splitk<FP8_SK_64x64_64>(m, n, k, 2, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
    } else if (n >= 2048) {
        // FF linear1 (N=4096), QKV (N=3072), Conv pw1 (N=2048)
        if (m <= 64)
            run_fp8_gemm<FP8_64x64_128_s3>(m, n, k, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
        else
            run_fp8_gemm<FP8_64x128_64_s6>(m, n, k, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
    } else if (n >= 1024) {
        // Pos proj, Attn out, Conv pw2 (N=1024)
        if (m <= 128)
            run_fp8_gemm<FP8_64x64_64_s6>(m, n, k, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
        else
            run_fp8_gemm<FP8_64x128_64_s6>(m, n, k, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
    } else {
        // Enc proj (N=640) and other small-N shapes
        run_fp8_gemm<FP8_64x64_128_s3>(m, n, k, X_fp8, ldA, W_fp8, ldB, Y, ldC, alpha_ptr, stream);
    }
}

} // namespace paraketto
