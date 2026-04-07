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

// Shared memory sizes
// sa: NR*NK halfs, sb: NC*NK halfs
// For store: NR*NC floats
constant constexpr size_t SA_SIZE = NR * NK;  // 2048 halfs
constant constexpr size_t SB_SIZE = NC * NK;  // 1024 halfs

// Helper: store simdgroup_float8x8 accumulators to device half* via threadgroup
template<typename DevPtr>
inline void store_accumulators(
    simdgroup_float8x8 mc[8],
    threadgroup float* shmem,
    DevPtr dst,
    uint r0, uint c0, uint M, uint N, uint ldc,
    ushort sg_r, ushort sg_c,
    ushort sgitg, ushort tiitg)
{
    // Layout: shmem[row * NC + col] for a NR×NC tile (row-major)
    // Each simdgroup stores its 32×16 sub-tile (4 row-blocks × 2 col-blocks of 8×8)
    threadgroup float* ts = shmem + sg_r * NC + sg_c;
    for (short i = 0; i < 8; i++) {
        // i%4 = row block (0..3), i/4 = col block (0..1)
        simdgroup_store(mc[i], ts + 8 * (i % 4) * NC + 8 * (i / 4),
                        NC, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads in first simdgroup write to device
    if (sgitg == 0) {
        ushort nr = min(uint(NR), M - r0);
        ushort nc = min(uint(NC), N - c0);
        for (uint j = tiitg; j < nc; j += 32) {
            for (uint i = 0; i < nr; i++) {
                dst[(r0 + i) * ldc + (c0 + j)] = half(shmem[i * NC + j]);
            }
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

    for (uint k = 0; k < p.K; k += NK) {
        // Load X[r0:r0+NR, k:k+NK] into sa with 8×8 tile layout
        // sa layout: for row r, col c within the tile:
        //   sa[block_r * NK * 8 + block_c * 64 + local_r * 8 + local_c]
        // where block_r = r/8, block_c = c/8, local_r = r%8, local_c = c%8
        // Simplified: sa[r * NK + c] in simple row-major — simdgroup_load
        //   with stride=NK will read 8 consecutive elements per row.
        // Actually let's just use simple row-major: sa[row * NK + col]
        for (ushort i = tiitg; i < NR * NK; i += 128) {
            ushort r = i / NK;
            ushort c = i % NK;
            uint gr = r0 + r;
            uint gc = k + c;
            sa[r * NK + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
        }

        // Load W[k:k+NK, c0:c0+NC] into sb
        // sb[row * NC + col] where row is K-index, col is N-index
        for (ushort i = tiitg; i < NK * NC; i += 128) {
            ushort r = i / NC;
            ushort c = i % NC;
            uint gr = k + r;
            uint gc = c0 + c;
            sb[r * NC + c] = (gr < p.K && gc < p.N) ? W[gr * p.ldb + gc] : half(0);
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
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * NK + ik, NK);
            }
            // Load B tiles: sb[ik, sg_c + 8*i], stride=NC, 2 tiles of 8 cols
            // B is [NK, NC], we want 8 rows (K) × 8 cols (N)
            // sb[ik * NC + sg_c + 8*i] with stride=NC
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], sb + ik * NC + sg_c + 8 * i, NC);
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

    for (uint k = 0; k < p.K; k += NK) {
        // Load X[r0:r0+NR, k:k+NK] into sa[NR, NK]
        for (ushort i = tiitg; i < NR * NK; i += 128) {
            ushort r = i / NK;
            ushort c = i % NK;
            uint gr = r0 + r;
            uint gc = k + c;
            sa[r * NK + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
        }

        // Load W^T: we want B[k_idx, n_idx] = W[n_idx, k_idx]
        // Store as sb[k_row * NC + n_col]
        for (ushort i = tiitg; i < NK * NC; i += 128) {
            ushort k_row = i / NC;
            ushort n_col = i % NC;
            uint gk = k + k_row;
            uint gn = c0 + n_col;
            sb[k_row * NC + n_col] = (gk < p.K && gn < p.N) ? W[gn * p.ldb + gk] : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];

            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * NK + ik, NK);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * NC + sg_c + 8 * i, NC);

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

    for (uint k = 0; k < p.K; k += NK) {
        for (ushort i = tiitg; i < NR * NK; i += 128) {
            ushort r = i / NK, c = i % NK;
            uint gr = r0 + r, gc = k + c;
            sa[r * NK + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
        }
        for (ushort i = tiitg; i < NK * NC; i += 128) {
            ushort r = i / NC, c = i % NC;
            uint gr = k + r, gc = c0 + c;
            sb[r * NC + c] = (gr < p.K && gc < p.N) ? bB[gr * p.ldb + gc] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8*i) * NK + ik, NK);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * NC + sg_c + 8*i, NC);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i%4], mb[i/4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

    for (uint k = 0; k < p.K; k += NK) {
        for (ushort i = tiitg; i < NR * NK; i += 128) {
            ushort r = i / NK, c = i % NK;
            uint gr = r0 + r, gc = k + c;
            sa[r * NK + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
        }
        // B^T: sb[k_row, n_col] = B[n_col, k_row]
        for (ushort i = tiitg; i < NK * NC; i += 128) {
            ushort k_row = i / NC, n_col = i % NC;
            uint gk = k + k_row, gn = c0 + n_col;
            sb[k_row * NC + n_col] = (gk < p.K && gn < p.N) ? bB[gn * p.ldb + gk] : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < NK; ik += 8) {
            simdgroup_half8x8 ma[4], mb[2];
            for (short i = 0; i < 4; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8*i) * NK + ik, NK);
            for (short i = 0; i < 2; i++)
                simdgroup_load(mb[i], sb + ik * NC + sg_c + 8*i, NC);
            for (short i = 0; i < 8; i++)
                simdgroup_multiply_accumulate(mc[i], ma[i%4], mb[i/4], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* temp = (threadgroup float*)shmem;
    store_accumulators(mc, temp, bC, r0, c0, p.M, p.N, p.ldc,
                       sg_r, sg_c, sgitg, tiitg);
}
