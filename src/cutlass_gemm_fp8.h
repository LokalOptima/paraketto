// cutlass_gemm_fp8.h — CUTLASS FP8 GEMM wrappers replacing cublasLt
//
// All shapes use SM89 TensorOp (mma.sync 16x8x32) with tile configs
// benchmarked to match or beat cublasLt on RTX 5070 Ti (SM120).
//
// TN layout (both operands K-contiguous):
//   Y[m,n] = alpha * X_fp8[m,k] @ W_fp8[n,k]^T
//
// NN weights are pre-transposed at init to [n,k] row-major so all
// FP8 GEMMs use the same TN kernel path.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace paraketto {

// Initialize CUTLASS FP8 workspace. Call once after cudaStreamCreate.
void cutlass_fp8_gemm_init(cudaStream_t stream);

// Free workspace.
void cutlass_fp8_gemm_free();

// ---------------------------------------------------------------------------
// Alpha (dequantization scale product) helpers
// ---------------------------------------------------------------------------

// Single alpha: *out = *a * *b  (all device pointers)
void fp8_compute_alpha(float* out, const float* a, const float* b,
                       cudaStream_t stream);

// Batch alpha: alphas[i] = w_scales[w_map(i)] * act_scales[i]
// Uses the fixed per-block mapping {1,2,0,5,6,7,8,3,4} for conformer GEMMs.
void fp8_compute_all_alphas(float* alphas, const float* w_scales,
                            const float* act_scales, int n_blocks,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// FP8 GEMM: Y[m,n] = alpha * X_fp8[m,k] @ W_fp8[n,k]^T
// ---------------------------------------------------------------------------
//
// X_fp8: [m,k] row-major (k contiguous), FP8 E4M3
// W_fp8: [n,k] row-major (k contiguous), FP8 E4M3
// alpha_ptr: device pointer to float scale product (w_scale * act_scale)
// Y: [m,n] row-major, FP16 output
void cutlass_fp8_gemm(cudaStream_t stream,
                      const uint8_t* X_fp8, int m, int k,
                      const uint8_t* W_fp8, int n,
                      const float* alpha_ptr, half* Y);

} // namespace paraketto
