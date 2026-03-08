// kernels_fp8.h — FP8 quantization kernels for Parakeet conformer
//
// Multi-block absmax + quantize for FP16 → FP8 E4M3 conversion.
// Used for both weight quantization at init and activation quantization at runtime.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// FP8 E4M3 quantization: multi-block absmax + scale + quantize
//   in:        [n] FP16 input
//   out:       [n] FP8 E4M3 output (as uint8_t to avoid cuda_fp8.h in header)
//   scale_out: [1] float — dequantization scale (max(|x|) / 448.0)
//   amax_buf:  [1] int — scratch for atomicMax (caller-allocated on GPU)
// ---------------------------------------------------------------------------
void quantize_absmax_fp16_to_fp8(const half* in, uint8_t* out, float* scale_out,
                                  int n, int* amax_buf, cudaStream_t stream);

// ---------------------------------------------------------------------------
// FP8 quantize with pre-computed scale (no absmax pass)
//   in:    [n] FP16 input
//   out:   [n] FP8 E4M3 output
//   scale: [1] float — pre-computed dequant scale (on GPU, from prior calibration)
// ---------------------------------------------------------------------------
void quantize_fp8_static(const half* in, uint8_t* out, const float* scale,
                           int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Broadcast bias add: x[i,j] += bias[j]  for i in [0,rows), j in [0,cols)
//   x: [rows, cols] row-major FP16 (modified in-place)
//   bias: [cols] FP16
// ---------------------------------------------------------------------------
void bias_add_row_fp16(half* x, const half* bias, int rows, int cols,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// In-place transpose of uint8_t matrix [rows, cols] → [cols, rows]
//   data: [rows, cols] row-major (overwritten with [cols, rows] row-major)
//   temp: scratch buffer, must be >= rows * cols bytes
// ---------------------------------------------------------------------------
void transpose_u8_inplace(uint8_t* data, int rows, int cols,
                           void* temp, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused FP8 output kernels — produce FP16 + FP8 simultaneously
//
// Each kernel is a copy of the FP16 variant that also writes FP8 E4M3 output
// using a pre-computed scale (from prior calibration pass). Used on runtime
// path (fp8_calibrated=true) to eliminate separate quantize_fp8_static calls.
// ---------------------------------------------------------------------------

// LayerNorm + FP8: y = LN(x) in FP16, fp8_out = quantize(LN(x)) in FP8
void layer_norm_fp8(const half* x, const half* gamma, const half* beta,
                    half* y, uint8_t* fp8_out, const float* fp8_scale,
                    int N, int D, float eps, cudaStream_t stream);

// Residual add + LayerNorm + FP8:
//   x += alpha * delta, ln_out = LN(x) in FP16, fp8_out = quantize(LN(x))
void residual_add_layer_norm_fp8(half* x, const half* delta, float alpha,
                                  const half* gamma, const half* beta,
                                  half* ln_out, uint8_t* fp8_out,
                                  const float* fp8_scale,
                                  int N, int D, float eps,
                                  cudaStream_t stream);

// SiLU in-place + FP8: x = silu(x) in FP16, fp8_out = quantize(silu(x))
void silu_inplace_fp8(half* x, uint8_t* fp8_out, const float* fp8_scale,
                      int n, cudaStream_t stream);

// Depthwise conv1d k=9 + SiLU + FP8: y = silu(conv(x)) in FP16, fp8_out = quantize
void depthwise_conv1d_k9_silu_fp8(const half* x, const half* w, const half* b,
                                   half* y, uint8_t* fp8_out,
                                   const float* fp8_scale,
                                   int T, int C, cudaStream_t stream);

// Transpose [A,B,C] -> [B,A,C] + FP8: out in FP16, fp8_out = quantize
void transpose_0213_fp8(const half* in, half* out, uint8_t* fp8_out,
                         const float* fp8_scale,
                         int A, int B, int C, cudaStream_t stream);

// Reshape [C,H,W] -> [H,C*W] + FP8: out in FP16, fp8_out = quantize
void reshape_chw_to_hcw_fp8(const half* in, half* out, uint8_t* fp8_out,
                             const float* fp8_scale,
                             int C, int H, int W, cudaStream_t stream);
