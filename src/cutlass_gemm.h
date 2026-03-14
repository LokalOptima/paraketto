// cutlass_gemm.h — CUTLASS FP16 GEMM wrappers replacing cuBLAS
//
// All shapes use SM80 TensorOp (mma.sync 16x8x16) with tile configs
// benchmarked to match or beat cuBLAS on RTX 5070 Ti (SM120).
//
// Row-major convention (matching ONNX/PyTorch):
//   NN: Y[m,n] = X[m,k] @ W[k,n]
//   NT: Y[m,n] = X[m,k] @ W[n,k]^T

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Initialize CUTLASS workspace. Call once after cudaStreamCreate.
void cutlass_gemm_init(cudaStream_t stream);

// Free workspace. Call before cudaStreamDestroy.
void cutlass_gemm_free();

// ---------------------------------------------------------------------------
// NN: Y[m,n] = X[m,k] @ W[k,n]
// ---------------------------------------------------------------------------
void cutlass_gemm_nn(cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y);

// NN + bias: Y[m,n] = X[m,k] @ W[k,n] + bias[n]
void cutlass_gemm_nn_bias(cudaStream_t stream,
                          const half* X, int m, int k,
                          const half* W, int n,
                          const half* bias, half* Y);

// ---------------------------------------------------------------------------
// NT: Y[m,n] = X[m,k] @ W[n,k]^T
// ---------------------------------------------------------------------------
void cutlass_gemm_nt(cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y);

// NT + bias: Y[m,n] = X[m,k] @ W[n,k]^T + bias[n]
void cutlass_gemm_nt_bias(cudaStream_t stream,
                          const half* X, int m, int k,
                          const half* W, int n,
                          const half* bias, half* Y);

// ---------------------------------------------------------------------------
// Batched strided GEMM (for multi-head attention)
// ---------------------------------------------------------------------------

// C[b,m,n] = A[b,m,k] @ B[b,k,n]
void cutlass_batched_gemm_nn(cudaStream_t stream,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC);

// C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
void cutlass_batched_gemm_nt(cudaStream_t stream,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC);

// NT with explicit leading dimensions and strides.
// Row-major: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
void cutlass_batched_gemm_nt_ex(cudaStream_t stream,
                                const half* A, int ldA, long long strideA,
                                const half* B, int ldB, long long strideB,
                                half* C, int ldC, long long strideC,
                                int batch, int m, int n, int k);
