// gemm.metal — Custom FP16 GEMM kernels for Parakeet conformer on Apple Silicon
//
// Uses simdgroup_matrix<half, 8, 8> for hardware-accelerated matrix multiply.
// Layout follows llama.cpp's proven pattern: 64×32 output tile, 4 simdgroups,
// threadgroup memory stores 8×8 sub-tiles with stride=8.
//
// GEMM conventions (row-major, matching gemm.h):
//   NN: Y[m,n] = X[m,k] @ W[k,n]
//   NT: Y[m,n] = X[m,k] @ W[n,k]^T

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// GEMM parameters
// ---------------------------------------------------------------------------
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
};

struct BatchedGemmParams {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
    uint batch;
    uint64_t strideA;
    uint64_t strideB;
    uint64_t strideC;
};

// ---------------------------------------------------------------------------
// Tile sizes: 64 output rows × 32 output cols, K-tile = 32
// 4 simdgroups (128 threads)
//   sg 0,1 → rows [0,32)   sg 2,3 → rows [32,64)
//   sg 0,2 → cols [0,16)   sg 1,3 → cols [16,32)
//
// Threadgroup memory layout (same as llama.cpp):
//   sa: 64 rows × 32 K-cols, stored as 8×8 tiles in row-major (stride=8)
//       Total: 64*32 = 2048 elements. Organized as 8 blocks of 8 tiles.
//       Access: sa[block * 64 + tile_within_block * 8 ... +8]
//   sb: 32 cols × 32 K-cols, same layout
//       Total: 32*32 = 1024 elements
// ---------------------------------------------------------------------------

constant constexpr short NR = 64;
constant constexpr short NC = 32;
constant constexpr short NK = 32;

// Threadgroup memory strides (same as tile dims — no padding needed since
// simdgroup_load reads 8 consecutive elements, not strided across banks)
constant constexpr short SA_STRIDE = NK;
constant constexpr short SB_STRIDE = NC;

// Shared memory sizes
constant constexpr size_t SA_SIZE = NR * SA_STRIDE;  // 64 × 32 = 2048 halfs
constant constexpr size_t SB_SIZE = NK * SB_STRIDE;  // 32 × 32 = 1024 halfs

// Store sub-tile size: each simdgroup owns 32 rows × 16 cols
constant constexpr short SG_NR = 32;
constant constexpr short SG_NC = 16;

// Helper: store simdgroup_float8x8 accumulators to device half* via threadgroup
//
// Each simdgroup stores its own 32×16 sub-tile independently:
//   - simdgroup_store 8 accumulators into threadgroup float[32×16]
//   - barrier within that simdgroup's region
//   - all 32 threads convert float→half and write to device
// All 4 simdgroups write in parallel (no idle threads).
template<typename DevPtr>
inline void store_accumulators(
    simdgroup_float8x8 mc[8],
    threadgroup float* shmem,
    DevPtr dst,
    uint r0, uint c0, uint M, uint N, uint ldc,
    ushort sg_r, ushort sg_c,
    ushort sgitg, ushort tiitg)
{
    // Each simdgroup gets its own 32×16 region in threadgroup memory
    // Layout: shmem[sgitg * SG_NR * SG_NC + row * SG_NC + col]
    threadgroup float* ts = shmem + sgitg * (SG_NR * SG_NC);

    // Store 8 accumulator tiles (4 row-blocks × 2 col-blocks of 8×8)
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], ts + 8 * (i % 4) * SG_NC + 8 * (i / 4),
                        SG_NC, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup's 32 threads write its 32×16 sub-tile to device
    // Thread lane handles one column, loops over rows
    ushort lane = tiitg % 32;
    uint abs_r = r0 + sg_r;
    uint abs_c = c0 + sg_c;
    ushort nr = (abs_r + SG_NR <= M) ? SG_NR : ((abs_r < M) ? (ushort)(M - abs_r) : 0);
    ushort nc = (abs_c + SG_NC <= N) ? SG_NC : ((abs_c < N) ? (ushort)(N - abs_c) : 0);

    for (ushort j = lane; j < nc; j += 32) {
        for (ushort i = 0; i < nr; i++) {
            dst[(abs_r + i) * ldc + (abs_c + j)] = half(ts[i * SG_NC + j]);
        }
    }
}

// Helper: store with fused SiLU activation (sigmoid(x) * x) applied in float
template<typename DevPtr>
inline void store_accumulators_silu(
    simdgroup_float8x8 mc[8],
    threadgroup float* shmem,
    DevPtr dst,
    uint r0, uint c0, uint M, uint N, uint ldc,
    ushort sg_r, ushort sg_c,
    ushort sgitg, ushort tiitg)
{
    threadgroup float* ts = shmem + sgitg * (SG_NR * SG_NC);
    for (short i = 0; i < 8; i++)
        simdgroup_store(mc[i], ts + 8 * (i % 4) * SG_NC + 8 * (i / 4),
                        SG_NC, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort lane = tiitg % 32;
    uint abs_r = r0 + sg_r;
    uint abs_c = c0 + sg_c;
    ushort nr = (abs_r + SG_NR <= M) ? SG_NR : ((abs_r < M) ? (ushort)(M - abs_r) : 0);
    ushort nc = (abs_c + SG_NC <= N) ? SG_NC : ((abs_c < N) ? (ushort)(N - abs_c) : 0);

    for (ushort j = lane; j < nc; j += 32) {
        for (ushort i = 0; i < nr; i++) {
            float v = ts[i * SG_NC + j];
            v = v / (1.0f + metal::exp(-v));  // SiLU = x * sigmoid(x)
            dst[(abs_r + i) * ldc + (abs_c + j)] = half(v);
        }
    }
}

// Helper: store with fused bias addition
template<typename DevPtr>
inline void store_accumulators_bias(
    simdgroup_float8x8 mc[8],
    threadgroup float* shmem,
    DevPtr dst,
    device const half* bias,
    uint r0, uint c0, uint M, uint N, uint ldc,
    ushort sg_r, ushort sg_c,
    ushort sgitg, ushort tiitg)
{
    threadgroup float* ts = shmem + sgitg * (SG_NR * SG_NC);
    for (short i = 0; i < 8; i++)
        simdgroup_store(mc[i], ts + 8 * (i % 4) * SG_NC + 8 * (i / 4),
                        SG_NC, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort lane = tiitg % 32;
    uint abs_r = r0 + sg_r;
    uint abs_c = c0 + sg_c;
    ushort nr = (abs_r + SG_NR <= M) ? SG_NR : ((abs_r < M) ? (ushort)(M - abs_r) : 0);
    ushort nc = (abs_c + SG_NC <= N) ? SG_NC : ((abs_c < N) ? (ushort)(N - abs_c) : 0);

    for (ushort j = lane; j < nc; j += 32) {
        float b = float(bias[abs_c + j]);
        for (ushort i = 0; i < nr; i++) {
            dst[(abs_r + i) * ldc + (abs_c + j)] = half(ts[i * SG_NC + j] + b);
        }
    }
}

// ---------------------------------------------------------------------------
// NN GEMM: Y[m,n] = X[m,k] @ W[k,n]
//   X: [M, K] row-major, W: [K, N] row-major, Y: [M, N] row-major
//
// Grid: (ceil(N/NC), ceil(M/NR), 1)  Threadgroup: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void gemm_nn_f16(
    device const half*     X     [[buffer(0)]],
    device const half*     W     [[buffer(1)]],
    device       half*     Y     [[buffer(2)]],
    constant GemmParams&   p     [[buffer(3)]],
    threadgroup  char*     shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // sa: [NR, NK] in half, sb: [NC, NK] in half
    // Stored with stride 8 for simdgroup_load: each 8×8 block is contiguous
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;

    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;

    // Each simdgroup handles a 32×16 sub-tile of the 64×32 output
    const ushort sg_r = (sgitg & 2) ? 32 : 0;   // 0 or 32
    const ushort sg_c = (sgitg & 1) ? 16 : 0;    // 0 or 16

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // Check if this threadgroup tile is fully within bounds (no boundary checks needed)
    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);

    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);

        // Load X[r0:r0+NR, k:k+NK] into sa
        // sa is row-major: sa[r * SA_STRIDE + c], NR rows × NK cols
        // 128 threads, NR*NK = 2048 elements, 16 per thread
        // Use half4 vectorized loads: 4 iterations of 4 elements
        if (full_m && full_k) {
            // Fast path: no bounds checks, vectorized loads
            // Each thread loads 16 halfs = 4 × half4
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4;
                ushort r = flat / NK;
                ushort c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) =
                    *(device const half4*)(X + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }

        // Load W[k:k+NK, c0:c0+NC] into sb
        // sb is row-major: sb[r * SB_STRIDE + c], NK rows × NC cols
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC / 4; i += 128) {
                ushort flat = i * 4;
                ushort r = flat / NC, c = flat % NC;
                *(threadgroup half4*)(sb + r * SB_STRIDE + c) =
                    *(device const half4*)(W + (k + r) * p.ldb + c0 + c);
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort r = i / NC, c = i % NC;
                uint gr = k + r, gc = c0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.K && gc < p.N) ? W[gr * p.ldb + gc] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply using simdgroup_matrix
        // For each K-step of 8:
        //   Load 4 tiles from sa (rows sg_r..sg_r+32, 8 K-cols) → ma[0..3]
        //   Load 2 tiles from sb (cols sg_c..sg_c+16, 8 K-cols) → mb[0..1]
        //   mc[i] += ma[i%4] * mb[i/4]  (output tiles)
        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            // Load A tiles: sa[sg_r + 8*i, ik], stride=NK, 4 tiles of 8 rows
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA_STRIDE + ik, SA_STRIDE);
            }
            // Load B tiles: sb[ik, sg_c + 8*i], stride=NC, 2 tiles of 8 cols
            // B is [NK, NC], we want 8 rows (K) × 8 cols (N)
            // sb[ik * NC + sg_c + 8*i] with stride=NC
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8 * i, SB_STRIDE);
            }

            // Accumulate: mc[row_tile * 2 + col_tile] += ma[row_tile] * mb[col_tile]
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], ma[i % 4], mb[i / 4], mc[i]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators(mc, temp, Y, r0, c0, p.M, p.N, p.ldc,
                       sg_r, sg_c, sgitg, tiitg);
}

// ---------------------------------------------------------------------------
// NN GEMM + fused SiLU: Y[m,n] = silu(X[m,k] @ W[k,n])
// ---------------------------------------------------------------------------
kernel void gemm_nn_silu_f16(
    device const half*     X     [[buffer(0)]],
    device const half*     W     [[buffer(1)]],
    device       half*     Y     [[buffer(2)]],
    constant GemmParams&   p     [[buffer(3)]],
    threadgroup  char*     shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;
    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;
    const ushort sg_r = (sgitg & 2) ? 32 : 0;
    const ushort sg_c = (sgitg & 1) ? 16 : 0;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    {
    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);
    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);
        if (full_m && full_k) {
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NK, c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) = *(device const half4*)(X + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK; uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NC, c = flat % NC;
                *(threadgroup half4*)(sb + r * SB_STRIDE + c) = *(device const half4*)(W + (k + r) * p.ldb + c0 + c);
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort r = i / NC, c = i % NC; uint gr = k + r, gc = c0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.K && gc < p.N) ? W[gr * p.ldb + gc] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA_STRIDE + ik, SA_STRIDE);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8 * i, SB_STRIDE);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i % 4], mb[i / 4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators_silu(mc, temp, Y, r0, c0, p.M, p.N, p.ldc,
                            sg_r, sg_c, sgitg, tiitg);
}

// ---------------------------------------------------------------------------
// NN GEMM + fused bias: Y[m,n] = X[m,k] @ W[k,n] + bias[n]
// ---------------------------------------------------------------------------
kernel void gemm_nn_bias_f16(
    device const half*     X     [[buffer(0)]],
    device const half*     W     [[buffer(1)]],
    device       half*     Y     [[buffer(2)]],
    constant GemmParams&   p     [[buffer(3)]],
    device const half*     bias  [[buffer(4)]],
    threadgroup  char*     shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;
    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;
    const ushort sg_r = (sgitg & 2) ? 32 : 0;
    const ushort sg_c = (sgitg & 1) ? 16 : 0;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    {
    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);
    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);
        if (full_m && full_k) {
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NK, c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) = *(device const half4*)(X + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK; uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NC, c = flat % NC;
                *(threadgroup half4*)(sb + r * SB_STRIDE + c) = *(device const half4*)(W + (k + r) * p.ldb + c0 + c);
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort r = i / NC, c = i % NC; uint gr = k + r, gc = c0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.K && gc < p.N) ? W[gr * p.ldb + gc] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA_STRIDE + ik, SA_STRIDE);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8 * i, SB_STRIDE);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i % 4], mb[i / 4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators_bias(mc, temp, Y, bias, r0, c0, p.M, p.N, p.ldc,
                            sg_r, sg_c, sgitg, tiitg);
}

// ---------------------------------------------------------------------------
// NT GEMM: Y[m,n] = X[m,k] @ W[n,k]^T
//   X: [M, K] row-major, W: [N, K] row-major, Y: [M, N] row-major
// ---------------------------------------------------------------------------
kernel void gemm_nt_f16(
    device const half*     X     [[buffer(0)]],
    device const half*     W     [[buffer(1)]],
    device       half*     Y     [[buffer(2)]],
    constant GemmParams&   p     [[buffer(3)]],
    threadgroup  char*     shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;

    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;

    const ushort sg_r = (sgitg & 2) ? 32 : 0;
    const ushort sg_c = (sgitg & 1) ? 16 : 0;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);

    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);

        // Load X[r0:r0+NR, k:k+NK] into sa[NR, NK]
        if (full_m && full_k) {
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4;
                ushort r = flat / NK, c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) =
                    *(device const half4*)(X + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }

        // Load W^T: sb[k_row, n_col] = W[n_col, k_row] (transpose, can't vectorize easily)
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort k_row = i / NC, n_col = i % NC;
                sb[k_row * SB_STRIDE + n_col] = W[(c0 + n_col) * p.ldb + k + k_row];
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort k_row = i / NC, n_col = i % NC;
                uint gk = k + k_row, gn = c0 + n_col;
                sb[k_row * SB_STRIDE + n_col] = (gk < p.K && gn < p.N) ? W[gn * p.ldb + gk] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];

            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA_STRIDE + ik, SA_STRIDE);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8 * i, SB_STRIDE);

            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i % 4], mb[i / 4], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators(mc, temp, Y, r0, c0, p.M, p.N, p.ldc,
                       sg_r, sg_c, sgitg, tiitg);
}

// ---------------------------------------------------------------------------
// GEMV: Y[1,n] = X[1,k] @ W[k,n]  (M=1, dot-product per output element)
// ---------------------------------------------------------------------------
kernel void gemv_nn_f16(
    device const half*     X [[buffer(0)]],
    device const half*     W [[buffer(1)]],
    device       half*     Y [[buffer(2)]],
    constant GemmParams&   p [[buffer(3)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]])
{
    uint n = tgpig;
    if (n >= p.N) return;

    float acc = 0.0f;
    for (uint k = tiitg; k < p.K; k += 256)
        acc += float(X[k]) * float(W[k * p.ldb + n]);

    acc = simd_sum(acc);

    threadgroup float s_buf[8];
    uint simd_id = tiitg / 32;
    uint lane = tiitg % 32;
    if (lane == 0) s_buf[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        acc = (lane < 8) ? s_buf[lane] : 0.0f;
        acc = simd_sum(acc);
        if (lane == 0) Y[n] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// Batched NN GEMM: C[b,m,n] = A[b,m,k] @ B[b,k,n]
//   Grid: (ceil(N/NC), ceil(M/NR), batch)
// ---------------------------------------------------------------------------
kernel void batched_gemm_nn_f16(
    device const half*          A     [[buffer(0)]],
    device const half*          B     [[buffer(1)]],
    device       half*          C     [[buffer(2)]],
    constant BatchedGemmParams& p     [[buffer(3)]],
    threadgroup  char*          shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    device const half* bA = A + tgpig.z * p.strideA;
    device const half* bB = B + tgpig.z * p.strideB;
    device       half* bC = C + tgpig.z * p.strideC;

    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;

    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;

    const ushort sg_r = (sgitg & 2) ? 32 : 0;
    const ushort sg_c = (sgitg & 1) ? 16 : 0;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    {
    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);
    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);
        if (full_m && full_k) {
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NK, c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) = *(device const half4*)(bA + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK; uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
            }
        }
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NC, c = flat % NC;
                *(threadgroup half4*)(sb + r * SB_STRIDE + c) = *(device const half4*)(bB + (k + r) * p.ldb + c0 + c);
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort r = i / NC, c = i % NC; uint gr = k + r, gc = c0 + c;
                sb[r * SB_STRIDE + c] = (gr < p.K && gc < p.N) ? bB[gr * p.ldb + gc] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8*i) * SA_STRIDE + ik, SA_STRIDE);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8*i, SB_STRIDE);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i%4], mb[i/4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators(mc, temp, bC, r0, c0, p.M, p.N, p.ldc,
                       sg_r, sg_c, sgitg, tiitg);
}

// ---------------------------------------------------------------------------
// Batched NT GEMM: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
// ---------------------------------------------------------------------------
kernel void batched_gemm_nt_f16(
    device const half*          A     [[buffer(0)]],
    device const half*          B     [[buffer(1)]],
    device       half*          C     [[buffer(2)]],
    constant BatchedGemmParams& p     [[buffer(3)]],
    threadgroup  char*          shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    device const half* bA = A + tgpig.z * p.strideA;
    device const half* bB = B + tgpig.z * p.strideB;
    device       half* bC = C + tgpig.z * p.strideC;

    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = sa + SA_SIZE;

    const uint r0 = tgpig.y * NR;
    const uint c0 = tgpig.x * NC;

    const ushort sg_r = (sgitg & 2) ? 32 : 0;
    const ushort sg_c = (sgitg & 1) ? 16 : 0;

    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    {
    const bool full_m = (r0 + NR <= p.M);
    const bool full_n = (c0 + NC <= p.N);
    for (uint k = 0; k < p.K; k += NK) {
        const bool full_k = (k + NK <= p.K);
        if (full_m && full_k) {
            for (ushort i = tiitg; i < NR * NK / 4; i += 128) {
                ushort flat = i * 4; ushort r = flat / NK, c = flat % NK;
                *(threadgroup half4*)(sa + r * SA_STRIDE + c) = *(device const half4*)(bA + (r0 + r) * p.lda + k + c);
            }
        } else {
            for (ushort i = tiitg; i < NR * NK; i += 128) {
                ushort r = i / NK, c = i % NK; uint gr = r0 + r, gc = k + c;
                sa[r * SA_STRIDE + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
            }
        }
        // B^T: sb[k_row, n_col] = B[n_col, k_row] (transpose, scalar loads)
        if (full_n && full_k) {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort k_row = i / NC, n_col = i % NC;
                sb[k_row * SB_STRIDE + n_col] = bB[(c0 + n_col) * p.ldb + k + k_row];
            }
        } else {
            for (ushort i = tiitg; i < NK * NC; i += 128) {
                ushort k_row = i / NC, n_col = i % NC;
                uint gk = k + k_row, gn = c0 + n_col;
                sb[k_row * SB_STRIDE + n_col] = (gk < p.K && gn < p.N) ? bB[gn * p.ldb + gk] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8*i) * SA_STRIDE + ik, SA_STRIDE);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * SB_STRIDE + sg_c + 8*i, SB_STRIDE);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i%4], mb[i/4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators(mc, temp, bC, r0, c0, p.M, p.N, p.ldc,
                       sg_r, sg_c, sgitg, tiitg);
}
