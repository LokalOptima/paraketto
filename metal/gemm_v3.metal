// gemm_v3.metal — Small-M GEMM variant with 32×64 tiles
//
// For skinny GEMMs (M < ~192), BM=32 gives more threadgroups and better
// GPU occupancy than BM=64, at the cost of fewer accumulators per simdgroup.
//
// BM=32: ceil(131/32)×64 = 320 threadgroups (10 waves on 32 cores)
// BM=64: ceil(131/64)×64 = 192 threadgroups (6 waves on 32 cores)
//
// Uses GemmParams, BatchedGemmParams from gemm.metal (concatenated before this).

// ---------------------------------------------------------------------------
// V3 tile parameters: 32×64×16
// ---------------------------------------------------------------------------

constant constexpr short BM3 = 32;
constant constexpr short BN3 = 64;
constant constexpr short BK3 = 16;

constant constexpr short WM3 = 2;
constant constexpr short WN3 = 2;

// Per-simdgroup: (BM3/WM3)×(BN3/WN3) = 16×32 in 8×8 fragments
constant constexpr short TM3 = BM3 / (8 * WM3);  // 2
constant constexpr short TN3 = BN3 / (8 * WN3);  // 4

constant constexpr short SA3_STRIDE = BK3 + 8;   // 24
constant constexpr short SB3_STRIDE = BN3 + 8;   // 72

constant constexpr size_t SA3_SIZE = BM3 * SA3_STRIDE;  // 768
constant constexpr size_t SB3_SIZE = BK3 * SB3_STRIDE;  // 1152

// Shmem: max(load=3840, store=8192) = 8192 bytes
constant constexpr size_t SHMEM3_TOTAL = BM3 * BN3 * sizeof(float);  // 8192

// A: 32×16 = 512 elements / 128 threads = 4 per thread (one half4)
// B: 16×64 = 1024 elements / 128 threads = 8 per thread (two half4, same as v2)

// ---------------------------------------------------------------------------
// NN GEMM v3: Y[m,n] = X[m,k] @ W[k,n]   (BM=32 variant)
//
// Grid: (ceil(N/64), ceil(M/32), 1)  Threadgroup: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void gemm_nn_v3_f16(
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
    threadgroup half* sb = sa + SA3_SIZE;

    const uint r0 = tgpig.y * BM3;
    const uint c0 = tgpig.x * BN3;

    const ushort sg_r = (sgitg / WN3) * (BM3 / WM3);  // 0 or 16
    const ushort sg_c = (sgitg % WN3) * (BN3 / WN3);  // 0 or 32

    // 8 accumulators: TM3 × TN3 = 2×4
    simdgroup_float8x8 mc[TM3 * TN3];
    for (short i = 0; i < TM3 * TN3; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const bool full_m = (r0 + BM3 <= p.M);
    const bool full_n = (c0 + BN3 <= p.N);

    // A load coords: 128 threads, 512 elements, 4 per thread (one half4)
    const ushort a_row = tiitg / (BK3 / 4);   // 0..31 (4 half4 per row)
    const ushort a_col = (tiitg % (BK3 / 4)) * 4;  // 0, 4, 8, 12

    // B load coords: same as v2 (8 per thread, two half4)
    const ushort b_vpr = BN3 / 8;  // 8 vectors per row
    const ushort b_row = tiitg / b_vpr;   // 0..15
    const ushort b_col = (tiitg % b_vpr) * 8;

    for (uint k = 0; k < p.K; k += BK3) {
        const bool full_k = (k + BK3 <= p.K);

        if (full_m && full_k) {
            *(threadgroup half4*)(sa + a_row * SA3_STRIDE + a_col) =
                *(device const half4*)(X + (r0 + a_row) * p.lda + k + a_col);
        } else {
            for (ushort i = tiitg; i < BM3 * BK3; i += 128) {
                ushort r = i / BK3, c = i % BK3;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA3_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }

        if (full_n && full_k) {
            *(threadgroup half4*)(sb + b_row * SB3_STRIDE + b_col) =
                *(device const half4*)(W + (k + b_row) * p.ldb + c0 + b_col);
            *(threadgroup half4*)(sb + b_row * SB3_STRIDE + b_col + 4) =
                *(device const half4*)(W + (k + b_row) * p.ldb + c0 + b_col + 4);
        } else {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort r = i / BN3, c = i % BN3;
                uint gr = k + r, gc = c0 + c;
                sb[r * SB3_STRIDE + c] = (gr < p.K && gc < p.N) ? W[gr * p.ldb + gc] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < BK3; ik += 8) {
            simdgroup_half8x8 ma[TM3], mb[TN3];
            for (short i = 0; i < TM3; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA3_STRIDE + ik, SA3_STRIDE);
            for (short i = 0; i < TN3; i++)
                simdgroup_load(mb[i], sb + ik * SB3_STRIDE + sg_c + 8 * i, SB3_STRIDE);
            for (short m = 0; m < TM3; m++)
                for (short n = 0; n < TN3; n++) {
                    short n_serp = (m % 2) ? (TN3 - 1 - n) : n;
                    simdgroup_multiply_accumulate(
                        mc[m * TN3 + n_serp], ma[m], mb[n_serp], mc[m * TN3 + n_serp]);
                }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store
    threadgroup float* ts = ((threadgroup float*)shmem) + sg_r * BN3 + sg_c;
    for (short i = 0; i < TM3 * TN3; i++) {
        short tm = i / TN3, tn = i % TN3;
        simdgroup_store(mc[i], ts + 8 * tm * BN3 + 8 * tn, BN3, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* all_ts = (threadgroup float*)shmem;
    ushort nr = min(uint(BM3), p.M - min(r0, p.M));
    ushort nc = min(uint(BN3), p.N - min(c0, p.N));
    for (ushort idx = tiitg; idx < nr * nc; idx += 128) {
        ushort r = idx / nc, c = idx % nc;
        Y[(r0 + r) * p.ldc + (c0 + c)] = half(all_ts[r * BN3 + c]);
    }
}

// ---------------------------------------------------------------------------
// NT GEMM v3: Y[m,n] = X[m,k] @ W[n,k]^T   (BM=32 variant)
// ---------------------------------------------------------------------------
kernel void gemm_nt_v3_f16(
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
    threadgroup half* sb = sa + SA3_SIZE;

    const uint r0 = tgpig.y * BM3;
    const uint c0 = tgpig.x * BN3;

    const ushort sg_r = (sgitg / WN3) * (BM3 / WM3);
    const ushort sg_c = (sgitg % WN3) * (BN3 / WN3);

    simdgroup_float8x8 mc[TM3 * TN3];
    for (short i = 0; i < TM3 * TN3; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const bool full_m = (r0 + BM3 <= p.M);
    const bool full_n = (c0 + BN3 <= p.N);

    const ushort a_row = tiitg / (BK3 / 4);
    const ushort a_col = (tiitg % (BK3 / 4)) * 4;

    for (uint k = 0; k < p.K; k += BK3) {
        const bool full_k = (k + BK3 <= p.K);

        if (full_m && full_k) {
            *(threadgroup half4*)(sa + a_row * SA3_STRIDE + a_col) =
                *(device const half4*)(X + (r0 + a_row) * p.lda + k + a_col);
        } else {
            for (ushort i = tiitg; i < BM3 * BK3; i += 128) {
                ushort r = i / BK3, c = i % BK3;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA3_STRIDE + c] = (gr < p.M && gc < p.K) ? X[gr * p.lda + gc] : half(0);
            }
        }

        // NT: sb[k_row, n_col] = W[n_col, k_row]
        if (full_n && full_k) {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort k_row = i / BN3, n_col = i % BN3;
                sb[k_row * SB3_STRIDE + n_col] = W[(c0 + n_col) * p.ldb + k + k_row];
            }
        } else {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort k_row = i / BN3, n_col = i % BN3;
                uint gk = k + k_row, gn = c0 + n_col;
                sb[k_row * SB3_STRIDE + n_col] = (gk < p.K && gn < p.N) ? W[gn * p.ldb + gk] : half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort ik = 0; ik < BK3; ik += 8) {
            simdgroup_half8x8 ma[TM3], mb[TN3];
            for (short i = 0; i < TM3; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA3_STRIDE + ik, SA3_STRIDE);
            for (short i = 0; i < TN3; i++)
                simdgroup_load(mb[i], sb + ik * SB3_STRIDE + sg_c + 8 * i, SB3_STRIDE);
            for (short m = 0; m < TM3; m++)
                for (short n = 0; n < TN3; n++) {
                    short n_serp = (m % 2) ? (TN3 - 1 - n) : n;
                    simdgroup_multiply_accumulate(
                        mc[m * TN3 + n_serp], ma[m], mb[n_serp], mc[m * TN3 + n_serp]);
                }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* ts = ((threadgroup float*)shmem) + sg_r * BN3 + sg_c;
    for (short i = 0; i < TM3 * TN3; i++) {
        short tm = i / TN3, tn = i % TN3;
        simdgroup_store(mc[i], ts + 8 * tm * BN3 + 8 * tn, BN3, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* all_ts = (threadgroup float*)shmem;
    ushort nr = min(uint(BM3), p.M - min(r0, p.M));
    ushort nc = min(uint(BN3), p.N - min(c0, p.N));
    for (ushort idx = tiitg; idx < nr * nc; idx += 128) {
        ushort r = idx / nc, c = idx % nc;
        Y[(r0 + r) * p.ldc + (c0 + c)] = half(all_ts[r * BN3 + c]);
    }
}

// ---------------------------------------------------------------------------
// Batched NN v3
// ---------------------------------------------------------------------------
kernel void batched_gemm_nn_v3_f16(
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
    threadgroup half* sb = sa + SA3_SIZE;

    const uint r0 = tgpig.y * BM3;
    const uint c0 = tgpig.x * BN3;
    const ushort sg_r = (sgitg / WN3) * (BM3 / WM3);
    const ushort sg_c = (sgitg % WN3) * (BN3 / WN3);

    simdgroup_float8x8 mc[TM3 * TN3];
    for (short i = 0; i < TM3 * TN3; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const bool full_m = (r0 + BM3 <= p.M);
    const bool full_n = (c0 + BN3 <= p.N);

    const ushort a_row = tiitg / (BK3 / 4);
    const ushort a_col = (tiitg % (BK3 / 4)) * 4;
    const ushort b_vpr = BN3 / 8;
    const ushort b_row = tiitg / b_vpr;
    const ushort b_col = (tiitg % b_vpr) * 8;

    for (uint k = 0; k < p.K; k += BK3) {
        const bool full_k = (k + BK3 <= p.K);
        if (full_m && full_k) {
            *(threadgroup half4*)(sa + a_row * SA3_STRIDE + a_col) =
                *(device const half4*)(bA + (r0 + a_row) * p.lda + k + a_col);
        } else {
            for (ushort i = tiitg; i < BM3 * BK3; i += 128) {
                ushort r = i / BK3, c = i % BK3;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA3_STRIDE + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
            }
        }
        if (full_n && full_k) {
            *(threadgroup half4*)(sb + b_row * SB3_STRIDE + b_col) =
                *(device const half4*)(bB + (k + b_row) * p.ldb + c0 + b_col);
            *(threadgroup half4*)(sb + b_row * SB3_STRIDE + b_col + 4) =
                *(device const half4*)(bB + (k + b_row) * p.ldb + c0 + b_col + 4);
        } else {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort r = i / BN3, c = i % BN3;
                uint gr = k + r, gc = c0 + c;
                sb[r * SB3_STRIDE + c] = (gr < p.K && gc < p.N) ? bB[gr * p.ldb + gc] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort ik = 0; ik < BK3; ik += 8) {
            simdgroup_half8x8 ma[TM3], mb[TN3];
            for (short i = 0; i < TM3; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA3_STRIDE + ik, SA3_STRIDE);
            for (short i = 0; i < TN3; i++)
                simdgroup_load(mb[i], sb + ik * SB3_STRIDE + sg_c + 8 * i, SB3_STRIDE);
            for (short m = 0; m < TM3; m++)
                for (short n = 0; n < TN3; n++) {
                    short n_serp = (m % 2) ? (TN3 - 1 - n) : n;
                    simdgroup_multiply_accumulate(mc[m * TN3 + n_serp], ma[m], mb[n_serp], mc[m * TN3 + n_serp]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* ts = ((threadgroup float*)shmem) + sg_r * BN3 + sg_c;
    for (short i = 0; i < TM3 * TN3; i++) {
        short tm = i / TN3, tn = i % TN3;
        simdgroup_store(mc[i], ts + 8 * tm * BN3 + 8 * tn, BN3, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* all_ts = (threadgroup float*)shmem;
    ushort nr = min(uint(BM3), p.M - min(r0, p.M));
    ushort nc = min(uint(BN3), p.N - min(c0, p.N));
    for (ushort idx = tiitg; idx < nr * nc; idx += 128) {
        ushort r = idx / nc, c = idx % nc;
        bC[(r0 + r) * p.ldc + (c0 + c)] = half(all_ts[r * BN3 + c]);
    }
}

// ---------------------------------------------------------------------------
// Batched NT v3
// ---------------------------------------------------------------------------
kernel void batched_gemm_nt_v3_f16(
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
    threadgroup half* sb = sa + SA3_SIZE;

    const uint r0 = tgpig.y * BM3;
    const uint c0 = tgpig.x * BN3;
    const ushort sg_r = (sgitg / WN3) * (BM3 / WM3);
    const ushort sg_c = (sgitg % WN3) * (BN3 / WN3);

    simdgroup_float8x8 mc[TM3 * TN3];
    for (short i = 0; i < TM3 * TN3; i++)
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const bool full_m = (r0 + BM3 <= p.M);
    const bool full_n = (c0 + BN3 <= p.N);

    const ushort a_row = tiitg / (BK3 / 4);
    const ushort a_col = (tiitg % (BK3 / 4)) * 4;

    for (uint k = 0; k < p.K; k += BK3) {
        const bool full_k = (k + BK3 <= p.K);
        if (full_m && full_k) {
            *(threadgroup half4*)(sa + a_row * SA3_STRIDE + a_col) =
                *(device const half4*)(bA + (r0 + a_row) * p.lda + k + a_col);
        } else {
            for (ushort i = tiitg; i < BM3 * BK3; i += 128) {
                ushort r = i / BK3, c = i % BK3;
                uint gr = r0 + r, gc = k + c;
                sa[r * SA3_STRIDE + c] = (gr < p.M && gc < p.K) ? bA[gr * p.lda + gc] : half(0);
            }
        }
        if (full_n && full_k) {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort k_row = i / BN3, n_col = i % BN3;
                sb[k_row * SB3_STRIDE + n_col] = bB[(c0 + n_col) * p.ldb + k + k_row];
            }
        } else {
            for (ushort i = tiitg; i < BK3 * BN3; i += 128) {
                ushort k_row = i / BN3, n_col = i % BN3;
                uint gk = k + k_row, gn = c0 + n_col;
                sb[k_row * SB3_STRIDE + n_col] = (gk < p.K && gn < p.N) ? bB[gn * p.ldb + gk] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort ik = 0; ik < BK3; ik += 8) {
            simdgroup_half8x8 ma[TM3], mb[TN3];
            for (short i = 0; i < TM3; i++)
                simdgroup_load(ma[i], sa + (sg_r + 8 * i) * SA3_STRIDE + ik, SA3_STRIDE);
            for (short i = 0; i < TN3; i++)
                simdgroup_load(mb[i], sb + ik * SB3_STRIDE + sg_c + 8 * i, SB3_STRIDE);
            for (short m = 0; m < TM3; m++)
                for (short n = 0; n < TN3; n++) {
                    short n_serp = (m % 2) ? (TN3 - 1 - n) : n;
                    simdgroup_multiply_accumulate(mc[m * TN3 + n_serp], ma[m], mb[n_serp], mc[m * TN3 + n_serp]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* ts = ((threadgroup float*)shmem) + sg_r * BN3 + sg_c;
    for (short i = 0; i < TM3 * TN3; i++) {
        short tm = i / TN3, tn = i % TN3;
        simdgroup_store(mc[i], ts + 8 * tm * BN3 + 8 * tn, BN3, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* all_ts = (threadgroup float*)shmem;
    ushort nr = min(uint(BM3), p.M - min(r0, p.M));
    ushort nc = min(uint(BN3), p.N - min(c0, p.N));
    for (ushort idx = tiitg; idx < nr * nc; idx += 128) {
        ushort r = idx / nc, c = idx % nc;
        bC[(r0 + r) * p.ldc + (c0 + c)] = half(all_ts[r * BN3 + c]);
    }
}
