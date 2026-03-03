// cutlass_gemm.h — Custom CUTLASS FP8 GEMM for SM120 (Blackwell GeForce)
//
// TN layout: A = RowMajor FP8 E4M3, B = ColumnMajor FP8 E4M3
// Output: RowMajor FP16, FP32 accumulation.
// alpha = act_scale * wt_scale (dequant folded into epilogue).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// Query maximum workspace size needed for any GEMM we'll run.
// Call once at init, allocate workspace, reuse for all GEMMs.
// ---------------------------------------------------------------------------
size_t cutlass_fp8_workspace_size(int max_M, int max_N, int max_K);

// ---------------------------------------------------------------------------
// FP8 GEMM: D[M,N] = alpha * A_fp8[M,K] @ B_fp8[N,K] (TN layout)
//
//   A_fp8:     [M, K] RowMajor, K contiguous (FP8 E4M3 as uint8_t)
//   B_fp8:     [N, K] RowMajor, K contiguous (FP8 E4M3 as uint8_t)
//              CRITICAL: SM120 CUTLASS reads B with K-contiguous access.
//              NN weights W[K,N] must be transposed to W^T[N,K] before calling.
//              NT weights W[N,K] can be passed directly.
//   D:         [M, N] RowMajor FP16 output
//   alpha:     dequant scale = act_scale * wt_scale
//   workspace: pre-allocated from cutlass_fp8_workspace_size()
//
// Returns 0 on success, non-zero on error.
// ---------------------------------------------------------------------------
int cutlass_fp8_gemm(int M, int N, int K,
                     const uint8_t* A_fp8, const uint8_t* B_fp8,
                     half* D,
                     float alpha,
                     cudaStream_t stream,
                     void* workspace, size_t workspace_size);
