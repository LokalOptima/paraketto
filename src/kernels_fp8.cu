// kernels_fp8.cu — FP8 quantization kernels for Parakeet conformer
//
// Multi-block absmax + quantize: FP16 → FP8 E4M3 with per-tensor scaling.
// Two-kernel approach: (1) multi-block absmax via atomicMax, (2) multi-block quantize.

#include "kernels_fp8.h"
#include "common.h"
#include <cuda_fp8.h>
#include <cfloat>

namespace paraketto {

static constexpr float FP8_E4M3_MAX = 448.0f;

// ---------------------------------------------------------------------------
// Kernel 1: Multi-block absmax reduction via atomicMax (float-as-int trick)
// Requires *amax_as_int to be zeroed before launch.
// ---------------------------------------------------------------------------
__global__ void absmax_reduce_kernel(const half* __restrict__ in,
                                      int* __restrict__ amax_as_int,
                                      int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = 0.0f;
    for (int i = gid; i < n; i += stride) {
        float v = fabsf(__half2float(in[i]));
        if (v > local_max) local_max = v;
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));

    // One atomicMax per warp (positive floats have same ordering as ints)
    if ((threadIdx.x & 31) == 0)
        atomicMax(amax_as_int, __float_as_int(local_max));
}

// ---------------------------------------------------------------------------
// Kernel 2: Multi-block quantize using previously computed amax
// Reads amax from amax_as_int (set by absmax_reduce_kernel), computes scale,
// writes scale to scale_out, and quantizes all elements.
// ---------------------------------------------------------------------------
__global__ void quantize_fp8_kernel(const half* __restrict__ in,
                                     uint8_t* __restrict__ out,
                                     const int* __restrict__ amax_as_int,
                                     float* __restrict__ scale_out,
                                     int n) {
    float amax = __int_as_float(*amax_as_int);
    float scale = (amax > 0.0f) ? (amax / FP8_E4M3_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Thread 0 writes the dequantization scale
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *scale_out = scale;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < n; i += stride)
        out[i] = __nv_cvt_float_to_fp8(__half2float(in[i]) * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

void quantize_absmax_fp16_to_fp8(const half* in, uint8_t* out, float* scale_out,
                                  int n, int* amax_buf, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 128);

    // Reset amax buffer to 0 (for atomicMax)
    CUDA_CHECK(cudaMemsetAsync(amax_buf, 0, sizeof(int), stream));

    // Phase 1: multi-block absmax reduction
    absmax_reduce_kernel<<<blocks, threads, 0, stream>>>(in, amax_buf, n);

    // Phase 2: multi-block quantize
    quantize_fp8_kernel<<<blocks, threads, 0, stream>>>(in, out, amax_buf, scale_out, n);
}

// ---------------------------------------------------------------------------
// Single-pass quantize with pre-computed scale (no absmax needed)
// Used after calibration when the activation scale is cached.
// ---------------------------------------------------------------------------
__global__ void quantize_fp8_static_kernel(const half* __restrict__ in,
                                            uint8_t* __restrict__ out,
                                            const float* __restrict__ scale_ptr,
                                            int n) {
    float inv_scale = 1.0f / (*scale_ptr);
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < n; i += stride)
        out[i] = __nv_cvt_float_to_fp8(__half2float(in[i]) * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

void quantize_fp8_static(const half* in, uint8_t* out, const float* scale,
                           int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 128);
    quantize_fp8_static_kernel<<<blocks, threads, 0, stream>>>(in, out, scale, n);
}

// ---------------------------------------------------------------------------
// Broadcast bias add: x[i,j] += bias[j]  (float32 intermediate for FP8 path)
// ---------------------------------------------------------------------------

__global__ void bias_add_row_kernel(half* __restrict__ x,
                                     const half* __restrict__ bias,
                                     int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int j = idx % cols;
        x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[j]));
    }
}

void bias_add_row_fp16(half* x, const half* bias, int rows, int cols,
                        cudaStream_t stream) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_row_kernel<<<blocks, threads, 0, stream>>>(x, bias, rows, cols);
}

// ---------------------------------------------------------------------------
// Transpose uint8_t matrix: src[rows,cols] → dst[cols,rows]
// ---------------------------------------------------------------------------

__global__ void transpose_u8_kernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        dst[c * rows + r] = src[idx];
    }
}

void transpose_u8_inplace(uint8_t* data, int rows, int cols,
                           void* temp, cudaStream_t stream) {
    int total = rows * cols;
    CUDA_CHECK(cudaMemcpyAsync(temp, data, total, cudaMemcpyDeviceToDevice, stream));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_u8_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)temp, data, rows, cols);
}

// =========================================================================
// Fused FP8 output kernels
// =========================================================================

// ---------------------------------------------------------------------------
// LayerNorm + FP8
// ---------------------------------------------------------------------------

__global__ void layer_norm_fp8_kernel(const half* __restrict__ x,
                                       const half* __restrict__ gamma,
                                       const half* __restrict__ beta,
                                       half* __restrict__ y,
                                       uint8_t* __restrict__ fp8_out,
                                       const float* __restrict__ fp8_scale,
                                       int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const half* xr = x + row * D;
    half* yr = y + row * D;
    uint8_t* fp8r = fp8_out + row * D;

    float sum = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sum += v;
        sum2 += v * v;
    }

    for (int mask = 16; mask > 0; mask >>= 1) {
        sum  += __shfl_xor_sync(0xffffffff, sum, mask);
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
    }

    __shared__ float s_sum[32], s_sum2[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) { s_sum[warp_id] = sum; s_sum2[warp_id] = sum2; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        sum  = (lane < nwarps) ? s_sum[lane] : 0.0f;
        sum2 = (lane < nwarps) ? s_sum2[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum  += __shfl_xor_sync(0xffffffff, sum, mask);
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = sum / D;
        float var = sum2 / D - s_mean * s_mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;
    float inv_scale = 1.0f / (*fp8_scale);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        float val = (v - mean) * inv_std * g + b;
        yr[i] = __float2half(val);
        fp8r[i] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

void layer_norm_fp8(const half* x, const half* gamma, const half* beta,
                    half* y, uint8_t* fp8_out, const float* fp8_scale,
                    int N, int D, float eps, cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    layer_norm_fp8_kernel<<<N, threads, 0, stream>>>(x, gamma, beta, y, fp8_out, fp8_scale, N, D, eps);
}

// ---------------------------------------------------------------------------
// Residual add + LayerNorm + FP8
// ---------------------------------------------------------------------------

__global__ void residual_add_layer_norm_fp8_kernel(half* __restrict__ x,
                                                    const half* __restrict__ delta,
                                                    float alpha,
                                                    const half* __restrict__ gamma,
                                                    const half* __restrict__ beta,
                                                    half* __restrict__ ln_out,
                                                    uint8_t* __restrict__ fp8_out,
                                                    const float* __restrict__ fp8_scale,
                                                    int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    half* xr = x + row * D;
    const half* dr = delta + row * D;
    half* yr = ln_out + row * D;
    uint8_t* fp8r = fp8_out + row * D;

    float sum = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]) + alpha * __half2float(dr[i]);
        xr[i] = __float2half(v);
        sum += v;
        sum2 += v * v;
    }

    for (int mask = 16; mask > 0; mask >>= 1) {
        sum  += __shfl_xor_sync(0xffffffff, sum, mask);
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
    }

    __shared__ float s_sum[32], s_sum2[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) { s_sum[warp_id] = sum; s_sum2[warp_id] = sum2; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        sum  = (lane < nwarps) ? s_sum[lane] : 0.0f;
        sum2 = (lane < nwarps) ? s_sum2[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum  += __shfl_xor_sync(0xffffffff, sum, mask);
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = sum / D;
        float var = sum2 / D - s_mean * s_mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;
    float inv_scale = 1.0f / (*fp8_scale);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        float val = (v - mean) * inv_std * g + b;
        yr[i] = __float2half(val);
        fp8r[i] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

void residual_add_layer_norm_fp8(half* x, const half* delta, float alpha,
                                  const half* gamma, const half* beta,
                                  half* ln_out, uint8_t* fp8_out,
                                  const float* fp8_scale,
                                  int N, int D, float eps,
                                  cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    residual_add_layer_norm_fp8_kernel<<<N, threads, 0, stream>>>(
        x, delta, alpha, gamma, beta, ln_out, fp8_out, fp8_scale, N, D, eps);
}

// ---------------------------------------------------------------------------
// SiLU in-place + FP8
// ---------------------------------------------------------------------------

__global__ void silu_inplace_fp8_kernel(half* __restrict__ x,
                                         uint8_t* __restrict__ fp8_out,
                                         const float* __restrict__ fp8_scale,
                                         int n) {
    float inv_scale = 1.0f / (*fp8_scale);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        float val = v / (1.0f + expf(-v));
        x[i] = __float2half(val);
        fp8_out[i] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

void silu_inplace_fp8(half* x, uint8_t* fp8_out, const float* fp8_scale,
                      int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_inplace_fp8_kernel<<<blocks, threads, 0, stream>>>(x, fp8_out, fp8_scale, n);
}

// ---------------------------------------------------------------------------
// Depthwise conv1d k=9 + SiLU + FP8
// ---------------------------------------------------------------------------

__global__ void depthwise_conv1d_k9_silu_fp8_kernel(const half* __restrict__ x,
                                                      const half* __restrict__ w,
                                                      const half* __restrict__ b,
                                                      half* __restrict__ y,
                                                      uint8_t* __restrict__ fp8_out,
                                                      const float* __restrict__ fp8_scale,
                                                      int T, int C) {
    float inv_scale = 1.0f / (*fp8_scale);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * C;
    if (idx < total) {
        int t = idx / C;
        int c = idx % C;

        float sum = 0.0f;
        const half* wc = w + c * 9;

        #pragma unroll
        for (int k = 0; k < 9; k++) {
            int ti = t + k - 4;
            if (ti >= 0 && ti < T)
                sum += __half2float(x[ti * C + c]) * __half2float(wc[k]);
        }

        if (b) sum += __half2float(b[c]);
        float val = sum / (1.0f + expf(-sum));
        y[idx] = __float2half(val);
        fp8_out[idx] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

void depthwise_conv1d_k9_silu_fp8(const half* x, const half* w, const half* b,
                                   half* y, uint8_t* fp8_out,
                                   const float* fp8_scale,
                                   int T, int C, cudaStream_t stream) {
    int total = T * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    depthwise_conv1d_k9_silu_fp8_kernel<<<blocks, threads, 0, stream>>>(
        x, w, b, y, fp8_out, fp8_scale, T, C);
}

// ---------------------------------------------------------------------------
// Transpose [A,B,C] -> [B,A,C] + FP8
// ---------------------------------------------------------------------------

__global__ void transpose_0213_fp8_kernel(const half* __restrict__ in,
                                           half* __restrict__ out,
                                           uint8_t* __restrict__ fp8_out,
                                           const float* __restrict__ fp8_scale,
                                           int A, int B, int C) {
    float inv_scale = 1.0f / (*fp8_scale);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx < total) {
        int a = idx / (B * C);
        int rem = idx % (B * C);
        int b = rem / C;
        int c = rem % C;
        int out_idx = b * A * C + a * C + c;
        float val = __half2float(in[idx]);
        out[out_idx] = __float2half(val);
        fp8_out[out_idx] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

void transpose_0213_fp8(const half* in, half* out, uint8_t* fp8_out,
                         const float* fp8_scale,
                         int A, int B, int C, cudaStream_t stream) {
    int total = A * B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_0213_fp8_kernel<<<blocks, threads, 0, stream>>>(
        in, out, fp8_out, fp8_scale, A, B, C);
}

// ---------------------------------------------------------------------------
// Reshape [C,H,W] -> [H,C*W] + FP8
// ---------------------------------------------------------------------------

__global__ void reshape_chw_to_hcw_fp8_kernel(const half* __restrict__ in,
                                               half* __restrict__ out,
                                               uint8_t* __restrict__ fp8_out,
                                               const float* __restrict__ fp8_scale,
                                               int C, int H, int W) {
    float inv_scale = 1.0f / (*fp8_scale);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;
    int h = idx / (C * W);
    int cw = idx % (C * W);
    int c = cw / W;
    int w = cw % W;
    float val = __half2float(in[c * H * W + h * W + w]);
    out[idx] = __float2half(val);
    fp8_out[idx] = __nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

void reshape_chw_to_hcw_fp8(const half* in, half* out, uint8_t* fp8_out,
                             const float* fp8_scale,
                             int C, int H, int W, cudaStream_t stream) {
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_chw_to_hcw_fp8_kernel<<<blocks, threads, 0, stream>>>(
        in, out, fp8_out, fp8_scale, C, H, W);
}

} // namespace paraketto
