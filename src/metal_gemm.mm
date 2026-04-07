// metal_gemm.mm — Metal GEMM dispatch for Parakeet conformer
//
// Dispatches to custom simdgroup_matrix GEMM kernels defined in metal/gemm.metal.
// For M=1 (decoder GEMV), dispatches to a specialized dot-product kernel.

#import <Metal/Metal.h>

#include "metal_gemm.h"
#include "metal_kernels.h"  // for metal_bias_add_fp16
#include "metal_context.h"
#include "metal_context_impl.h"
#include "common_metal.h"

// Match metal/gemm.metal struct layout
struct GemmParams {
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
};

struct BatchedGemmParams {
    uint32_t M, N, K;
    uint32_t lda, ldb, ldc;
    uint32_t batch;
    uint64_t strideA, strideB, strideC;
};

static inline id<MTLComputeCommandEncoder> enc(MetalEncoder e) {
    return (__bridge id<MTLComputeCommandEncoder>)e;
}
static inline id<MTLBuffer> buf(MetalBuffer b) {
    return (__bridge id<MTLBuffer>)b;
}

// Threadgroup memory: sa[NK*NR] + sb[NC*NK] in half = (32*64 + 32*32)*2 = 6144 bytes
// For bounds-checked store path, we also need NR*NC floats = 64*32*4 = 8192 bytes
// Total: max(6144, 8192) — but we reuse shmem for store, so 8192 covers both.
static constexpr uint NR = 64;
static constexpr uint NC = 32;
static constexpr uint NK = 32;
static constexpr size_t SHMEM_SIZE = (NK * NR + NC * NK) * sizeof(__fp16);
static constexpr size_t SHMEM_STORE = NR * NC * sizeof(float);
static constexpr size_t SHMEM_TOTAL = (SHMEM_SIZE > SHMEM_STORE) ? SHMEM_SIZE : SHMEM_STORE;

void metal_gemm_init(MetalContext& ctx) {
    // No persistent workspace needed (unlike CUTLASS split-K)
}

void metal_gemm_free() {}

// ---------------------------------------------------------------------------
// NN GEMM
// ---------------------------------------------------------------------------

void metal_gemm_nn(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                   size_t X_off, int m, int k,
                   size_t W_off, int n, size_t Y_off) {
    auto e = enc(encoder);

    if (m == 1) {
        // GEMV path: one threadgroup per output element
        auto pso = ctx.impl->get_pipeline("gemv_nn_f16");
        [e setComputePipelineState:pso];
        [e setBuffer:buf(pool) offset:X_off atIndex:0];
        [e setBuffer:buf(pool) offset:W_off atIndex:1];
        [e setBuffer:buf(pool) offset:Y_off atIndex:2];
        GemmParams p = {(uint32_t)m, (uint32_t)n, (uint32_t)k,
                        (uint32_t)k, (uint32_t)n, (uint32_t)n};
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }

    auto pso = ctx.impl->get_pipeline("gemm_nn_f16");
    [e setComputePipelineState:pso];
    [e setBuffer:buf(pool) offset:X_off atIndex:0];
    [e setBuffer:buf(pool) offset:W_off atIndex:1];
    [e setBuffer:buf(pool) offset:Y_off atIndex:2];
    GemmParams p = {(uint32_t)m, (uint32_t)n, (uint32_t)k,
                    (uint32_t)k, (uint32_t)n, (uint32_t)n};
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:SHMEM_TOTAL atIndex:0];

    uint grid_x = (n + NC - 1) / NC;
    uint grid_y = (m + NR - 1) / NR;
    [e dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void metal_gemm_nn_bias(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                        size_t X_off, int m, int k,
                        size_t W_off, int n,
                        size_t bias_off, size_t Y_off) {
    metal_gemm_nn(ctx, encoder, pool, X_off, m, k, W_off, n, Y_off);
    metal_bias_add_fp16(ctx, encoder, pool, Y_off, bias_off, m, n);
}

// ---------------------------------------------------------------------------
// NT GEMM
// ---------------------------------------------------------------------------

void metal_gemm_nt(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                   size_t X_off, int m, int k,
                   size_t W_off, int n, size_t Y_off) {
    auto e = enc(encoder);

    // No GEMV special case for NT (decoder uses NN)
    auto pso = ctx.impl->get_pipeline("gemm_nt_f16");
    [e setComputePipelineState:pso];
    [e setBuffer:buf(pool) offset:X_off atIndex:0];
    [e setBuffer:buf(pool) offset:W_off atIndex:1];
    [e setBuffer:buf(pool) offset:Y_off atIndex:2];
    GemmParams p = {(uint32_t)m, (uint32_t)n, (uint32_t)k,
                    (uint32_t)k, (uint32_t)k, (uint32_t)n};
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:SHMEM_TOTAL atIndex:0];

    uint grid_x = (n + NC - 1) / NC;
    uint grid_y = (m + NR - 1) / NR;
    [e dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void metal_gemm_nt_bias(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                        size_t X_off, int m, int k,
                        size_t W_off, int n,
                        size_t bias_off, size_t Y_off) {
    metal_gemm_nt(ctx, encoder, pool, X_off, m, k, W_off, n, Y_off);
    metal_bias_add_fp16(ctx, encoder, pool, Y_off, bias_off, m, n);
}

// ---------------------------------------------------------------------------
// Batched NN GEMM
// ---------------------------------------------------------------------------

void metal_batched_gemm_nn(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                           size_t A_off, size_t B_off, size_t C_off,
                           int batch, int m, int n, int k,
                           int64_t strideA, int64_t strideB, int64_t strideC) {
    auto e = enc(encoder);
    auto pso = ctx.impl->get_pipeline("batched_gemm_nn_f16");
    [e setComputePipelineState:pso];
    [e setBuffer:buf(pool) offset:A_off atIndex:0];
    [e setBuffer:buf(pool) offset:B_off atIndex:1];
    [e setBuffer:buf(pool) offset:C_off atIndex:2];
    BatchedGemmParams p = {
        (uint32_t)m, (uint32_t)n, (uint32_t)k,
        (uint32_t)k, (uint32_t)n, (uint32_t)n,
        (uint32_t)batch,
        (uint64_t)strideA, (uint64_t)strideB, (uint64_t)strideC
    };
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:SHMEM_TOTAL atIndex:0];

    uint grid_x = (n + NC - 1) / NC;
    uint grid_y = (m + NR - 1) / NR;
    [e dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// ---------------------------------------------------------------------------
// Batched NT GEMM
// ---------------------------------------------------------------------------

void metal_batched_gemm_nt(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                           size_t A_off, size_t B_off, size_t C_off,
                           int batch, int m, int n, int k,
                           int64_t strideA, int64_t strideB, int64_t strideC) {
    auto e = enc(encoder);
    auto pso = ctx.impl->get_pipeline("batched_gemm_nt_f16");
    [e setComputePipelineState:pso];
    [e setBuffer:buf(pool) offset:A_off atIndex:0];
    [e setBuffer:buf(pool) offset:B_off atIndex:1];
    [e setBuffer:buf(pool) offset:C_off atIndex:2];
    BatchedGemmParams p = {
        (uint32_t)m, (uint32_t)n, (uint32_t)k,
        (uint32_t)k, (uint32_t)k, (uint32_t)n,
        (uint32_t)batch,
        (uint64_t)strideA, (uint64_t)strideB, (uint64_t)strideC
    };
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:SHMEM_TOTAL atIndex:0];

    uint grid_x = (n + NC - 1) / NC;
    uint grid_y = (m + NR - 1) / NR;
    [e dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

// ---------------------------------------------------------------------------
// Batched NT with explicit leading dimensions
// ---------------------------------------------------------------------------

void metal_batched_gemm_nt_ex(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                              size_t A_off, int ldA, int64_t strideA,
                              size_t B_off, int ldB, int64_t strideB,
                              size_t C_off, int ldC, int64_t strideC,
                              int batch, int m, int n, int k) {
    auto e = enc(encoder);
    auto pso = ctx.impl->get_pipeline("batched_gemm_nt_f16");
    [e setComputePipelineState:pso];
    [e setBuffer:buf(pool) offset:A_off atIndex:0];
    [e setBuffer:buf(pool) offset:B_off atIndex:1];
    [e setBuffer:buf(pool) offset:C_off atIndex:2];
    BatchedGemmParams p = {
        (uint32_t)m, (uint32_t)n, (uint32_t)k,
        (uint32_t)ldA, (uint32_t)ldB, (uint32_t)ldC,
        (uint32_t)batch,
        (uint64_t)strideA, (uint64_t)strideB, (uint64_t)strideC
    };
    [e setBytes:&p length:sizeof(p) atIndex:3];
    [e setThreadgroupMemoryLength:SHMEM_TOTAL atIndex:0];

    uint grid_x = (n + NC - 1) / NC;
    uint grid_y = (m + NR - 1) / NR;
    [e dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}
