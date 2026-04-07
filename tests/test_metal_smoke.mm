// test_metal_smoke.mm — Smoke test for Metal backend
//
// Verifies: device creation, shader compilation, kernel dispatch, readback.
// Run: clang++ -std=c++17 -O3 -fobjc-arc -Isrc -framework Metal -framework Foundation \
//      tests/test_metal_smoke.mm src/metal_context.mm -o tests/test_metal_smoke && ./tests/test_metal_smoke

#import <Metal/Metal.h>
#include "metal_context.h"
#include "metal_context_impl.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Read the shader source at compile time would require the metal compiler.
// Instead, include it as a raw string for this test.
static const char* SHADER_SRC = R"(
#include <metal_stdlib>
using namespace metal;

kernel void silu_inplace_kernel(
    device half*    x [[buffer(0)]],
    constant uint&  n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = float(x[gid]);
    x[gid] = half(v / (1.0f + exp(-v)));
}

kernel void layer_norm_kernel(
    device const half*  x      [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],
    device const half*  beta   [[buffer(2)]],
    device       half*  y      [[buffer(3)]],
    constant     uint&  N      [[buffer(4)]],
    constant     uint&  D      [[buffer(5)]],
    constant     float& eps    [[buffer(6)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    uint row = tgid;
    if (row >= N) return;

    device const half* xr = x + row * D;
    device       half* yr = y + row * D;

    float sum = 0.0f, sum2 = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        sum  += v;
        sum2 += v * v;
    }
    sum  = simd_sum(sum);
    sum2 = simd_sum(sum2);

    threadgroup float s_sum[32], s_sum2[32];
    if (lane == 0) { s_sum[simd_id] = sum; s_sum2[simd_id] = sum2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nwarps = tg_size / 32;
        sum  = (lane < nwarps) ? s_sum[lane]  : 0.0f;
        sum2 = (lane < nwarps) ? s_sum2[lane] : 0.0f;
        sum  = simd_sum(sum);
        sum2 = simd_sum(sum2);
    }
    threadgroup float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean    = sum / float(D);
        float var = sum2 / float(D) - s_mean * s_mean;
        s_inv_std = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean    = s_mean;
    float inv_std = s_inv_std;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        float g = float(gamma[i]);
        float b = float(beta[i]);
        yr[i] = half((v - mean) * inv_std * g + b);
    }
}
)";

static float randf() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static bool test_silu(MetalContext& ctx) {
    printf("  SiLU in-place... ");
    const int N = 1024;
    auto pso = ctx.impl->get_pipeline("silu_inplace_kernel");

    id<MTLBuffer> buf = [ctx.impl->device
        newBufferWithLength:N * sizeof(__fp16)
                    options:MTLResourceStorageModeShared];

    // Fill with test data
    __fp16* ptr = (__fp16*)buf.contents;
    std::vector<float> ref(N);
    for (int i = 0; i < N; i++) {
        float v = randf();
        ptr[i] = (__fp16)v;
        ref[i] = v / (1.0f + expf(-v));  // SiLU reference
    }

    // Dispatch
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    uint n = N;
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Verify
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        float got = (float)ptr[i];
        float err = fabsf(got - ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("max_err=%.6f %s\n", max_err, max_err < 0.01f ? "PASS" : "FAIL");
    return max_err < 0.01f;
}

static bool test_layer_norm(MetalContext& ctx) {
    printf("  LayerNorm... ");
    const int N = 4, D = 128;
    auto pso = ctx.impl->get_pipeline("layer_norm_kernel");

    size_t buf_size = (N * D + D + D + N * D) * sizeof(__fp16);
    id<MTLBuffer> pool = [ctx.impl->device
        newBufferWithLength:buf_size
                    options:MTLResourceStorageModeShared];

    __fp16* base = (__fp16*)pool.contents;
    __fp16* x     = base;
    __fp16* gamma = base + N * D;
    __fp16* beta  = gamma + D;
    __fp16* y     = beta + D;

    // Fill test data
    srand(42);
    for (int i = 0; i < N * D; i++) x[i] = (__fp16)randf();
    for (int i = 0; i < D; i++) {
        gamma[i] = (__fp16)(1.0f + randf() * 0.1f);
        beta[i]  = (__fp16)(randf() * 0.1f);
    }

    // CPU reference
    std::vector<float> ref(N * D);
    float eps = 1e-5f;
    for (int row = 0; row < N; row++) {
        float sum = 0, sum2 = 0;
        for (int i = 0; i < D; i++) {
            float v = (float)x[row * D + i];
            sum += v; sum2 += v * v;
        }
        float mean = sum / D;
        float var = sum2 / D - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < D; i++) {
            float v = (float)x[row * D + i];
            float g = (float)gamma[i];
            float b = (float)beta[i];
            ref[row * D + i] = (v - mean) * inv_std * g + b;
        }
    }

    // Dispatch
    uint threads = 128;  // D=128, fits in one warp group
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:pool offset:0                                    atIndex:0];  // x
    [enc setBuffer:pool offset:(N * D) * sizeof(__fp16)             atIndex:1];  // gamma
    [enc setBuffer:pool offset:(N * D + D) * sizeof(__fp16)         atIndex:2];  // beta
    [enc setBuffer:pool offset:(N * D + D + D) * sizeof(__fp16)     atIndex:3];  // y
    uint nn = N, dd = D;
    [enc setBytes:&nn  length:sizeof(nn)  atIndex:4];
    [enc setBytes:&dd  length:sizeof(dd)  atIndex:5];
    [enc setBytes:&eps length:sizeof(eps) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Verify
    float max_err = 0;
    for (int i = 0; i < N * D; i++) {
        float got = (float)y[i];
        float err = fabsf(got - ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("max_err=%.6f %s\n", max_err, max_err < 0.05f ? "PASS" : "FAIL");
    return max_err < 0.05f;
}

int main() {
    printf("Metal smoke test\n");
    printf("================\n");

    // 1. Create context
    MetalContext ctx;

    // 2. Compile shaders
    printf("Compiling shaders... ");
    ctx.load_shaders(SHADER_SRC, "smoke_test");
    printf("OK\n");

    printf("Max threadgroup memory: %zu bytes\n", ctx.max_threadgroup_memory());

    // 3. Run kernel tests
    bool ok = true;
    ok &= test_silu(ctx);
    ok &= test_layer_norm(ctx);

    printf("================\n");
    printf("%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
