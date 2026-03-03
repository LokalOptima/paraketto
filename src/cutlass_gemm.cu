// cutlass_gemm.cu — Custom CUTLASS FP8 GEMM kernel for SM120
//
// Non-block-scaled FP8 E4M3 × FP8 E4M3 → FP16 with FP32 accumulation.
// TN layout (A=RowMajor, B=ColumnMajor), cluster 1×1×1, tile 128×128×128.
//
// CRITICAL: SM120 CUTLASS reads B with K-contiguous memory access regardless
// of the ColumnMajor layout specification. This means B_fp8 must be stored
// as [N,K] row-major (K contiguous), NOT as [K,N] row-major.
// → Weight matrices W[K,N] must be transposed to W^T[N,K] before calling.

#include "cutlass_gemm.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cstdio>
#include <cstdlib>

using namespace cute;

// -------------------------------------------------------------------------
// Kernel type definition — matches vLLM SM120 config
// -------------------------------------------------------------------------

using ElementA_  = cutlass::float_e4m3_t;
using ElementB_  = cutlass::float_e4m3_t;
using ElementC_  = cutlass::half_t;    // beta=0, C never read
using ElementD_  = cutlass::half_t;    // FP16 output
using ElementAcc = float;
using ElementCmp = float;

using LayoutA_   = cutlass::layout::RowMajor;
using LayoutB_   = cutlass::layout::ColumnMajor;
using LayoutC_   = cutlass::layout::RowMajor;
using LayoutD_   = cutlass::layout::RowMajor;

static constexpr int kAlignAB = 16;   // 128-bit / 8-bit = 16 elements
static constexpr int kAlignCD = 8;    // 128-bit / 16-bit = 8 elements

using Arch    = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassTensorOp;

using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Build epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCmp,
    ElementC_, LayoutC_, kAlignCD,
    ElementD_, LayoutD_, kAlignCD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Build mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OpClass,
    ElementA_, LayoutA_, kAlignAB,
    ElementB_, LayoutB_, kAlignAB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Assemble kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

size_t cutlass_fp8_workspace_size(int max_M, int max_N, int max_K) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(max_M, max_K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(max_N, max_K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(max_M, max_N, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {max_M, max_N, max_K, 1},
        {nullptr, stride_A, nullptr, stride_B},
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr, stride_D}
    };

    return Gemm::get_workspace_size(args);
}

int cutlass_fp8_gemm(int M, int N, int K,
                     const uint8_t* A_fp8, const uint8_t* B_fp8,
                     half* D,
                     float alpha,
                     cudaStream_t stream,
                     void* workspace, size_t workspace_size) {

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    // D[M,N] = alpha * A_fp8[M,K] @ B_fp8[N,K]
    //
    // A_fp8: RowMajor [M,K] — K contiguous (activation, quantized at runtime)
    // B_fp8: RowMajor [N,K] — K contiguous (weight, pre-transposed from [K,N])
    // D:     RowMajor [M,N] — N contiguous (output)

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

    auto A_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(A_fp8);
    auto B_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(B_fp8);
    auto D_ptr = reinterpret_cast<cutlass::half_t*>(D);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {A_ptr, stride_A, B_ptr, stride_B},
        {{alpha, 0.0f}, nullptr, stride_D, D_ptr, stride_D}
    };

    Gemm gemm_op;

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 GEMM can_implement failed: M=%d N=%d K=%d status=%d\n",
                M, N, K, (int)status);
        return (int)status;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 GEMM initialize failed: status=%d\n", (int)status);
        return (int)status;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 GEMM run failed: status=%d\n", (int)status);
        return (int)status;
    }

    return 0;
}
