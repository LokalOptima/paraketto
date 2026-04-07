// metal_gemm.h — Metal GEMM dispatch interface (mirrors gemm.h)
//
// Row-major convention (matching ONNX/PyTorch):
//   NN: Y[m,n] = X[m,k] @ W[k,n]
//   NT: Y[m,n] = X[m,k] @ W[n,k]^T

#pragma once

#include <cstddef>
#include <cstdint>

struct MetalContext;
using MetalEncoder = void*;
using MetalBuffer = void*;

void metal_gemm_init(MetalContext& ctx);
void metal_gemm_free();

// NN: Y[m,n] = X[m,k] @ W[k,n]
void metal_gemm_nn(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                   size_t X_off, int m, int k,
                   size_t W_off, int n, size_t Y_off);

// NN + bias: Y[m,n] = X[m,k] @ W[k,n] + bias[n]
void metal_gemm_nn_bias(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                        size_t X_off, int m, int k,
                        size_t W_off, int n,
                        size_t bias_off, size_t Y_off);

// NT: Y[m,n] = X[m,k] @ W[n,k]^T
void metal_gemm_nt(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                   size_t X_off, int m, int k,
                   size_t W_off, int n, size_t Y_off);

// NT + bias: Y[m,n] = X[m,k] @ W[n,k]^T + bias[n]
void metal_gemm_nt_bias(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                        size_t X_off, int m, int k,
                        size_t W_off, int n,
                        size_t bias_off, size_t Y_off);

// Batched strided NN: C[b,m,n] = A[b,m,k] @ B[b,k,n]
void metal_batched_gemm_nn(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                           size_t A_off, size_t B_off, size_t C_off,
                           int batch, int m, int n, int k,
                           int64_t strideA, int64_t strideB, int64_t strideC);

// Batched strided NT: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
void metal_batched_gemm_nt(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                           size_t A_off, size_t B_off, size_t C_off,
                           int batch, int m, int n, int k,
                           int64_t strideA, int64_t strideB, int64_t strideC);

// NT with explicit leading dimensions and strides
void metal_batched_gemm_nt_ex(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                              size_t A_off, int ldA, int64_t strideA,
                              size_t B_off, int ldB, int64_t strideB,
                              size_t C_off, int ldC, int64_t strideC,
                              int batch, int m, int n, int k);
