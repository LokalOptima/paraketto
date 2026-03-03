// kernels_fp8.cu — FP8 quantization kernels for Parakeet conformer
//
// Multi-block absmax + quantize: FP16 → FP8 E4M3 with per-tensor scaling.
// Two-kernel approach: (1) multi-block absmax via atomicMax, (2) multi-block quantize.

#include "kernels_fp8.h"
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#endif

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
// Broadcast bias add: x[i,j] += bias[j]
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
