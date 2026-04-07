// kernels.metal — Custom Metal compute shaders for Parakeet conformer
//
// All kernels use FP16 I/O with FP32 accumulation where needed.
// Ported from src/kernels.cu (CUDA) to Metal Shading Language.
//
// CUDA → Metal mapping:
//   __global__             → kernel
//   __shared__             → threadgroup
//   blockIdx.x             → tgid (threadgroup_position_in_grid)
//   threadIdx.x            → tid  (thread_position_in_threadgroup)
//   blockDim.x             → tg_size (threads_per_threadgroup)
//   __syncthreads()        → threadgroup_barrier(mem_flags::mem_threadgroup)
//   __shfl_xor_sync(m,v,l) → simd_shuffle_xor(v, l)
//   atomicAdd(&x, v)       → atomic_fetch_add_explicit(&x, v, memory_order_relaxed)

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Mel filterbank entry (matches kernels.h MelFBEntry)
// ---------------------------------------------------------------------------
struct MelFBEntry {
    uint16_t freq;
    uint16_t mel;
    float    weight;
};

// ---------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//   x, y:        [N, D]
//   gamma, beta:  [D]
//   One threadgroup per row, FP32 accumulation.
// ---------------------------------------------------------------------------
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

    // Compute mean and variance with SIMD-level reduction
    float sum = 0.0f, sum2 = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        sum  += v;
        sum2 += v * v;
    }

    // SIMD (warp) reduction
    sum  = simd_sum(sum);
    sum2 = simd_sum(sum2);

    // Block reduction via threadgroup memory (for tg_size > 32)
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

    // Broadcast mean and inv_std
    threadgroup float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean    = sum / float(D);
        float var = sum2 / float(D) - s_mean * s_mean;
        s_inv_std = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean    = s_mean;
    float inv_std = s_inv_std;

    // Normalize
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        float g = float(gamma[i]);
        float b = float(beta[i]);
        yr[i] = half((v - mean) * inv_std * g + b);
    }
}

// ---------------------------------------------------------------------------
// Fused residual add + LayerNorm
//   x_out[row] = x[row] + alpha * delta[row]   (writes back to x)
//   ln_out[row] = LN(x_out[row])               (normalized output)
// ---------------------------------------------------------------------------
kernel void residual_add_layer_norm_kernel(
    device       half*  x      [[buffer(0)]],
    device const half*  delta  [[buffer(1)]],
    constant     float& alpha  [[buffer(2)]],
    device const half*  gamma  [[buffer(3)]],
    device const half*  beta   [[buffer(4)]],
    device       half*  ln_out [[buffer(5)]],
    constant     uint&  N      [[buffer(6)]],
    constant     uint&  D      [[buffer(7)]],
    constant     float& eps    [[buffer(8)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    uint row = tgid;
    if (row >= N) return;

    device half*       xr = x      + row * D;
    device const half* dr = delta  + row * D;
    device half*       yr = ln_out + row * D;

    // Pass 1: update x and compute mean/var
    float sum = 0.0f, sum2 = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]) + alpha * float(dr[i]);
        xr[i] = half(v);
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

    // Pass 2: normalize
    for (uint i = tid; i < D; i += tg_size) {
        float v = float(xr[i]);
        float g = float(gamma[i]);
        float b = float(beta[i]);
        yr[i] = half((v - mean) * inv_std * g + b);
    }
}

// ---------------------------------------------------------------------------
// SiLU in-place: x = x * sigmoid(x)
// ---------------------------------------------------------------------------
kernel void silu_inplace_kernel(
    device half*    x [[buffer(0)]],
    constant uint&  n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = float(x[gid]);
    x[gid] = half(v / (1.0f + exp(-v)));
}

// ---------------------------------------------------------------------------
// GLU (Gated Linear Unit): y = x[:, :D] * sigmoid(x[:, D:])
//   Input [N, 2D], output [N, D]
// ---------------------------------------------------------------------------
kernel void glu_kernel(
    device const half*  x [[buffer(0)]],
    device       half*  y [[buffer(1)]],
    constant     uint&  N [[buffer(2)]],
    constant     uint&  D [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * D;
    if (gid >= total) return;
    uint row = gid / D;
    uint col = gid % D;
    float val  = float(x[row * 2 * D + col]);
    float gate = float(x[row * 2 * D + D + col]);
    y[gid] = half(val * (1.0f / (1.0f + exp(-gate))));
}

// ---------------------------------------------------------------------------
// Add + ReLU: y = max(a + b, 0)
// ---------------------------------------------------------------------------
kernel void add_relu_kernel(
    device const half*  a [[buffer(0)]],
    device const half*  b [[buffer(1)]],
    device       half*  y [[buffer(2)]],
    constant     uint&  n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float sum = float(a[gid]) + float(b[gid]);
    y[gid] = half(max(sum, 0.0f));
}

// ---------------------------------------------------------------------------
// Dual argmax: token = argmax(logits[:vocab_size]),
//              step  = argmax(logits[vocab_size:])
//   One threadgroup, 256 threads with SIMD reduction.
// ---------------------------------------------------------------------------
kernel void dual_argmax_kernel(
    device const half*  logits     [[buffer(0)]],
    device       int*   out        [[buffer(1)]],
    constant     uint&  vocab_size [[buffer(2)]],
    constant     uint&  total      [[buffer(3)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    // Token argmax: logits[0..vocab_size)
    float best_val_tok = -INFINITY;
    int   best_idx_tok = 0;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        float v = float(logits[i]);
        if (v > best_val_tok) { best_val_tok = v; best_idx_tok = int(i); }
    }

    // Step argmax: logits[vocab_size..total)
    float best_val_dur = -INFINITY;
    int   best_idx_dur = 0;
    for (uint i = vocab_size + tid; i < total; i += tg_size) {
        float v = float(logits[i]);
        if (v > best_val_dur) { best_val_dur = v; best_idx_dur = int(i - vocab_size); }
    }

    // SIMD reduction for token
    threadgroup float s_val[32];
    threadgroup int   s_idx[32];

    // -- token reduction --
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_xor(best_val_tok, offset);
        int   other_idx = simd_shuffle_xor(best_idx_tok, offset);
        if (other_val > best_val_tok) {
            best_val_tok = other_val;
            best_idx_tok = other_idx;
        }
    }
    if (lane == 0) { s_val[simd_id] = best_val_tok; s_idx[simd_id] = best_idx_tok; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint nwarps = tg_size / 32;
        float v = (lane < nwarps) ? s_val[lane] : -INFINITY;
        int   idx = (lane < nwarps) ? s_idx[lane] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_xor(v, offset);
            int   oi = simd_shuffle_xor(idx, offset);
            if (ov > v) { v = ov; idx = oi; }
        }
        if (lane == 0) out[0] = idx;
    }

    // -- duration reduction --
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_xor(best_val_dur, offset);
        int   other_idx = simd_shuffle_xor(best_idx_dur, offset);
        if (other_val > best_val_dur) {
            best_val_dur = other_val;
            best_idx_dur = other_idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { s_val[simd_id] = best_val_dur; s_idx[simd_id] = best_idx_dur; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint nwarps = tg_size / 32;
        float v = (lane < nwarps) ? s_val[lane] : -INFINITY;
        int   idx = (lane < nwarps) ? s_idx[lane] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_xor(v, offset);
            int   oi = simd_shuffle_xor(idx, offset);
            if (ov > v) { v = ov; idx = oi; }
        }
        if (lane == 0) out[1] = idx;
    }
}

// ---------------------------------------------------------------------------
// Depthwise conv 1D, kernel=9 + SiLU fused
//   x: [T, C], w: [C, 1, 9], b: [C], y: [T, C]
// ---------------------------------------------------------------------------
kernel void depthwise_conv1d_k9_silu_kernel(
    device const half*  x [[buffer(0)]],
    device const half*  w [[buffer(1)]],
    device const half*  b [[buffer(2)]],
    device       half*  y [[buffer(3)]],
    constant     uint&  T [[buffer(4)]],
    constant     uint&  C [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = T * C;
    if (gid >= total) return;

    uint t = gid / C;
    uint c = gid % C;

    float acc = float(b[c]);
    for (int k = 0; k < 9; k++) {
        int src_t = int(t) + k - 4;  // padding = 4
        if (src_t >= 0 && src_t < int(T)) {
            acc += float(x[uint(src_t) * C + c]) * float(w[c * 9 + uint(k)]);
        }
    }
    // Fused SiLU
    y[gid] = half(acc / (1.0f + exp(-acc)));
}

// ---------------------------------------------------------------------------
// Fused LSTM cell (single step)
//   gates: [1, 4*H]  (pre-computed: W_ih @ x + W_hh @ h_prev + bias)
//   c_prev: [1, H]
//   h_out, c_out: [1, H]
// ---------------------------------------------------------------------------
kernel void lstm_cell_kernel(
    device const half*  gates  [[buffer(0)]],
    device const half*  c_prev [[buffer(1)]],
    device       half*  h_out  [[buffer(2)]],
    device       half*  c_out  [[buffer(3)]],
    constant     uint&  H      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= H) return;

    float i_gate = float(gates[gid]);          // input gate
    float o_gate = float(gates[H + gid]);      // output gate
    float f_gate = float(gates[2 * H + gid]);  // forget gate
    float g_gate = float(gates[3 * H + gid]);  // cell gate

    float i = 1.0f / (1.0f + exp(-i_gate));
    float o = 1.0f / (1.0f + exp(-o_gate));
    float f = 1.0f / (1.0f + exp(-f_gate));
    float g = tanh(g_gate);

    float c = f * float(c_prev[gid]) + i * g;
    float h = o * tanh(c);

    c_out[gid] = half(c);
    h_out[gid] = half(h);
}

// ---------------------------------------------------------------------------
// Embed + concat: out[0:D] = table[idx], out[D:2D] = h[0:D]
// ---------------------------------------------------------------------------
kernel void embed_concat_kernel(
    device const half*  table [[buffer(0)]],
    constant     uint&  idx   [[buffer(1)]],
    device const half*  h     [[buffer(2)]],
    device       half*  out   [[buffer(3)]],
    constant     uint&  D     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < D) {
        out[gid] = table[idx * D + gid];
    } else if (gid < 2 * D) {
        out[gid] = h[gid - D];
    }
}

// ---------------------------------------------------------------------------
// Concat two D-length vectors: out[0:D] = a, out[D:2D] = b
// ---------------------------------------------------------------------------
kernel void concat_vectors_kernel(
    device const half*  a   [[buffer(0)]],
    device const half*  b   [[buffer(1)]],
    device       half*  out [[buffer(2)]],
    constant     uint&  D   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < D) {
        out[gid] = a[gid];
    } else if (gid < 2 * D) {
        out[gid] = b[gid - D];
    }
}

// ---------------------------------------------------------------------------
// FP32 -> FP16 cast
// ---------------------------------------------------------------------------
kernel void cast_fp32_to_fp16_kernel(
    device const float* x [[buffer(0)]],
    device       half*  y [[buffer(1)]],
    constant     uint&  n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    y[gid] = half(x[gid]);
}

// ---------------------------------------------------------------------------
// Transpose 2D: [M, N] -> [N, M]
//   Tiled with threadgroup memory for coalesced access.
// ---------------------------------------------------------------------------
constant constexpr uint TILE_DIM = 32;

kernel void transpose_kernel(
    device const half*  x  [[buffer(0)]],
    device       half*  y  [[buffer(1)]],
    constant     uint&  M  [[buffer(2)]],
    constant     uint&  N  [[buffer(3)]],
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint2 tid    [[thread_position_in_threadgroup]])
{
    threadgroup half tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    // Read from x[M, N] into tile (coalesced reads)
    uint x_col = tgid.x * TILE_DIM + tid.x;
    uint x_row = tgid.y * TILE_DIM + tid.y;

    // Each thread handles 4 rows (threadgroup is 32x8)
    for (uint j = 0; j < TILE_DIM; j += 8) {
        uint r = x_row + j;
        if (r < M && x_col < N) {
            tile[tid.y + j][tid.x] = x[r * N + x_col];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write from tile to y[N, M] (coalesced writes)
    uint y_col = tgid.y * TILE_DIM + tid.x;
    uint y_row = tgid.x * TILE_DIM + tid.y;

    for (uint j = 0; j < TILE_DIM; j += 8) {
        uint r = y_row + j;
        if (r < N && y_col < M) {
            y[r * M + y_col] = tile[tid.x][tid.y + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Residual add: y = a + alpha * b
// ---------------------------------------------------------------------------
kernel void residual_add_kernel(
    device const half*  a     [[buffer(0)]],
    device const half*  b     [[buffer(1)]],
    device       half*  y     [[buffer(2)]],
    constant     uint&  n     [[buffer(3)]],
    constant     float& alpha [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    y[gid] = half(float(a[gid]) + alpha * float(b[gid]));
}

// ---------------------------------------------------------------------------
// Fused skew + scale + softmax for relative position attention
//   content_scores: [heads, T, T]
//   pos_scores_raw: [heads, T, 2T-1]
//   output:         [heads, T, T]
// ---------------------------------------------------------------------------
kernel void fused_score_softmax_kernel(
    device const half*  content  [[buffer(0)]],
    device const half*  pos_raw  [[buffer(1)]],
    device       half*  output   [[buffer(2)]],
    constant     uint&  T        [[buffer(3)]],
    constant     float& scale    [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    // One threadgroup per (head, row) pair
    uint head = tgid / T;
    uint t    = tgid % T;

    device const half* c_row = content + (head * T + t) * T;
    device const half* p_row = pos_raw + (head * T + t) * (2 * T - 1);
    device       half* o_row = output  + (head * T + t) * T;

    // Pass 1: compute scores and find max
    threadgroup float s_max_arr[32];
    float local_max = -INFINITY;
    for (uint j = tid; j < T; j += tg_size) {
        float score = (float(c_row[j]) + float(p_row[j + T - 1 - t])) * scale;
        local_max = max(local_max, score);
    }
    local_max = simd_max(local_max);
    if (lane == 0) s_max_arr[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nw = tg_size / 32;
        float v = (lane < nw) ? s_max_arr[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) s_max_arr[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = s_max_arr[0];

    // Pass 2: compute exp(score - max) and sum
    threadgroup float s_sum_arr[32];
    float local_sum = 0.0f;
    for (uint j = tid; j < T; j += tg_size) {
        float score = (float(c_row[j]) + float(p_row[j + T - 1 - t])) * scale;
        float e = exp(score - row_max);
        o_row[j] = half(e);  // store temporary
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) s_sum_arr[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nw = tg_size / 32;
        float v = (lane < nw) ? s_sum_arr[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) s_sum_arr[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / s_sum_arr[0];

    // Pass 3: normalize
    for (uint j = tid; j < T; j += tg_size) {
        o_row[j] = half(float(o_row[j]) * inv_sum);
    }
}

// ---------------------------------------------------------------------------
// Conv2D: general 2D convolution (NCHW format)
// ---------------------------------------------------------------------------
kernel void conv2d_kernel(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    device       half*  output_buf [[buffer(3)]],
    constant     uint&  C_in    [[buffer(4)]],
    constant     uint&  H_in    [[buffer(5)]],
    constant     uint&  W_in    [[buffer(6)]],
    constant     uint&  C_out   [[buffer(7)]],
    constant     uint&  H_out   [[buffer(8)]],
    constant     uint&  W_out   [[buffer(9)]],
    constant     uint&  kH      [[buffer(10)]],
    constant     uint&  kW      [[buffer(11)]],
    constant     uint&  stride  [[buffer(12)]],
    constant     uint&  pad     [[buffer(13)]],
    constant     uint&  groups      [[buffer(14)]],
    constant     uint&  c_per_group [[buffer(15)]],
    constant     uint&  has_bias    [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = C_out * H_out * W_out;
    if (gid >= total) return;

    uint w_out = gid % W_out;
    uint h_out = (gid / W_out) % H_out;
    uint c_out = gid / (H_out * W_out);

    uint g = c_out / (C_out / groups);

    float acc = has_bias ? float(bias[c_out]) : 0.0f;

    for (uint ci = 0; ci < c_per_group; ci++) {
        uint c_in_idx = g * c_per_group + ci;
        for (uint kh = 0; kh < kH; kh++) {
            for (uint kw = 0; kw < kW; kw++) {
                int ih = int(h_out * stride + kh) - int(pad);
                int iw = int(w_out * stride + kw) - int(pad);
                if (ih >= 0 && ih < int(H_in) && iw >= 0 && iw < int(W_in)) {
                    float x_val = float(input[c_in_idx * H_in * W_in +
                                               uint(ih) * W_in + uint(iw)]);
                    uint w_idx = c_out * (c_per_group * kH * kW) +
                                 ci * (kH * kW) + kh * kW + kw;
                    float w_val = float(weight[w_idx]);
                    acc += x_val * w_val;
                }
            }
        }
    }
    output_buf[gid] = half(acc);
}

// ---------------------------------------------------------------------------
// im2col for 2D convolution (NCHW format)
// ---------------------------------------------------------------------------
kernel void im2col_2d_kernel(
    device const half*  input  [[buffer(0)]],
    device       half*  col    [[buffer(1)]],
    constant     uint&  C_in   [[buffer(2)]],
    constant     uint&  H_in   [[buffer(3)]],
    constant     uint&  W_in   [[buffer(4)]],
    constant     uint&  kH     [[buffer(5)]],
    constant     uint&  kW     [[buffer(6)]],
    constant     uint&  stride [[buffer(7)]],
    constant     uint&  pad    [[buffer(8)]],
    constant     uint&  H_out  [[buffer(9)]],
    constant     uint&  W_out  [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = C_in * kH * kW * H_out * W_out;
    if (gid >= total) return;

    uint spatial = H_out * W_out;
    uint kernel_size = kH * kW;

    uint w_out = gid % W_out;
    uint h_out = (gid / W_out) % H_out;
    uint kw    = (gid / spatial) % kW;
    uint kh    = (gid / (spatial * kW)) % kH;
    uint c     = gid / (spatial * kernel_size);

    int ih = int(h_out * stride + kh) - int(pad);
    int iw = int(w_out * stride + kw) - int(pad);

    float val = 0.0f;
    if (ih >= 0 && ih < int(H_in) && iw >= 0 && iw < int(W_in)) {
        val = float(input[c * H_in * W_in + uint(ih) * W_in + uint(iw)]);
    }
    col[gid] = half(val);
}

// ---------------------------------------------------------------------------
// Per-channel bias + ReLU for NCHW data (in-place)
// ---------------------------------------------------------------------------
kernel void bias_relu_nchw_kernel(
    device       half*  x       [[buffer(0)]],
    device const half*  bias    [[buffer(1)]],
    constant     uint&  C       [[buffer(2)]],
    constant     uint&  spatial [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = C * spatial;
    if (gid >= total) return;
    uint c = gid / spatial;
    float v = float(x[gid]) + float(bias[c]);
    x[gid] = half(max(v, 0.0f));
}

// ---------------------------------------------------------------------------
// Row-broadcast bias add: x[i,j] += bias[j]
// ---------------------------------------------------------------------------
kernel void bias_add_kernel(
    device       half*  x    [[buffer(0)]],
    device const half*  bias [[buffer(1)]],
    constant     uint&  rows [[buffer(2)]],
    constant     uint&  cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = rows * cols;
    if (gid >= total) return;
    uint col = gid % cols;
    x[gid] = half(float(x[gid]) + float(bias[col]));
}

// ---------------------------------------------------------------------------
// Reshape [C, H, W] -> [H, C*W]
// ---------------------------------------------------------------------------
kernel void reshape_chw_to_hcw_kernel(
    device const half*  in  [[buffer(0)]],
    device       half*  out [[buffer(1)]],
    constant     uint&  C   [[buffer(2)]],
    constant     uint&  H   [[buffer(3)]],
    constant     uint&  W   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = C * H * W;
    if (gid >= total) return;
    uint w = gid % W;
    uint h = (gid / W) % H;
    uint c = gid / (H * W);
    out[h * C * W + c * W + w] = in[gid];
}

// ---------------------------------------------------------------------------
// Generate sinusoidal position encoding on GPU
//   output: [(2*T-1), d_model]
// ---------------------------------------------------------------------------
kernel void generate_pos_encoding_kernel(
    device       half*  output  [[buffer(0)]],
    constant     uint&  T       [[buffer(1)]],
    constant     uint&  d_model [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = (2 * T - 1) * (d_model / 2);
    if (gid >= total) return;

    uint pair = gid % (d_model / 2);
    uint p    = gid / (d_model / 2);

    float pos = float(int(T) - 1 - int(p));
    float dim_scale = pow(10000.0f, 2.0f * float(pair) / float(d_model));
    float angle = pos / dim_scale;

    uint row = p * d_model;
    output[row + 2 * pair]     = half(sin(angle));
    output[row + 2 * pair + 1] = half(cos(angle));
}

// ---------------------------------------------------------------------------
// Transpose [A, B, C] -> [B, A, C]  (swap first two dims)
// ---------------------------------------------------------------------------
kernel void transpose_0213_kernel(
    device const half*  in  [[buffer(0)]],
    device       half*  out [[buffer(1)]],
    constant     uint&  A   [[buffer(2)]],
    constant     uint&  B   [[buffer(3)]],
    constant     uint&  C   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = A * B * C;
    if (gid >= total) return;
    uint c = gid % C;
    uint b = (gid / C) % B;
    uint a = gid / (B * C);
    out[b * A * C + a * C + c] = in[gid];
}

// ---------------------------------------------------------------------------
// Fused split + transpose + pos_bias
//   in: [T, 3*D] -> q_u, q_v (with bias), K, V in [heads, T, head_dim]
// ---------------------------------------------------------------------------
kernel void split_transpose_qkv_bias_kernel(
    device const half*  in      [[buffer(0)]],
    device const half*  bias_u  [[buffer(1)]],
    device const half*  bias_v  [[buffer(2)]],
    device       half*  q_u     [[buffer(3)]],
    device       half*  q_v     [[buffer(4)]],
    device       half*  k       [[buffer(5)]],
    device       half*  v       [[buffer(6)]],
    constant     uint&  T        [[buffer(7)]],
    constant     uint&  heads    [[buffer(8)]],
    constant     uint&  head_dim [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint D = heads * head_dim;
    uint total = T * 3 * D;
    if (gid >= total) return;

    uint col   = gid % (3 * D);
    uint t     = gid / (3 * D);
    uint qkv   = col / D;        // 0=Q, 1=K, 2=V
    uint d     = col % D;
    uint h     = d / head_dim;
    uint hd    = d % head_dim;

    float val = float(in[gid]);
    uint out_idx = h * T * head_dim + t * head_dim + hd;

    if (qkv == 0) {
        q_u[out_idx] = half(val + float(bias_u[h * head_dim + hd]));
        q_v[out_idx] = half(val + float(bias_v[h * head_dim + hd]));
    } else if (qkv == 1) {
        k[out_idx] = half(val);
    } else {
        v[out_idx] = half(val);
    }
}

// ---------------------------------------------------------------------------
// Fused FFT-512 + mel filterbank + log
//   frames:  [n_frames, 512] real windowed audio frames (float32)
//   mel_out: [n_frames, 128] log-mel spectrogram (float32)
//   mel_fb:  [N_MEL_FB_ENTRIES] sparse filterbank entries
//   One threadgroup per frame, 256 threads.
// ---------------------------------------------------------------------------

// 9-bit bit reversal (replacing CUDA's __brev() >> 23)
inline uint bit_reverse_9(uint x) {
    x = ((x & 0x1F0) >> 5) | ((x & 0x00F) << 5) | (x & 0x010);
    x = ((x & 0x18C) >> 2) | ((x & 0x063) << 2) | (x & 0x010);
    x = ((x & 0x14A) >> 1) | ((x & 0x0A5) << 1) | (x & 0x010);
    return x & 0x1FF;
}

kernel void fft512_mel_log_kernel(
    device const float*      frames  [[buffer(0)]],
    device       float*      mel_out [[buffer(1)]],
    constant     MelFBEntry* mel_fb  [[buffer(2)]],
    constant     uint&       n_frames      [[buffer(3)]],
    constant     uint&       n_mel_entries [[buffer(4)]],
    constant     uint&       n_mels        [[buffer(5)]],
    constant     float&      log_eps       [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= n_frames) return;

    threadgroup float sr[512], si[512];

    // 1. Bit-reversal load
    device const float* in = frames + tgid * 512;
    uint i0 = tid, i1 = tid + 256;
    uint br0 = bit_reverse_9(i0);
    uint br1 = bit_reverse_9(i1);
    sr[br0] = in[i0]; si[br0] = 0.0f;
    sr[br1] = in[i1]; si[br1] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 9 butterfly stages
    for (uint s = 0; s < 9; s++) {
        uint size = 1u << (s + 1);
        uint half_size = size >> 1;
        uint group = tid / half_size;
        uint k = tid % half_size;
        uint a = group * size + k;
        uint b = a + half_size;

        float angle = -6.283185307179586f * float(k) / float(size);
        float wr = cos(angle);
        float wi = sin(angle);

        float tr = wr * sr[b] - wi * si[b];
        float ti = wr * si[b] + wi * sr[b];
        float ar = sr[a], ai = si[a];

        sr[a] = ar + tr; si[a] = ai + ti;
        sr[b] = ar - tr; si[b] = ai - ti;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Power spectrum -> sr[0..256]
    float power_tid = sr[tid] * sr[tid] + si[tid] * si[tid];
    float power_256 = 0.0f;
    if (tid == 0) power_256 = sr[256] * sr[256] + si[256] * si[256];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sr[tid] = power_tid;
    if (tid == 0) sr[256] = power_256;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Mel filterbank: scatter accumulation in threadgroup memory
    // Reuse si[0..n_mels-1] as mel accumulators
    if (tid < n_mels) si[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Serial accumulation by thread 0 (mel filterbank is sparse, ~504 entries)
    // This avoids needing atomic_float on threadgroup memory which Metal
    // doesn't support for float types in threadgroup address space.
    if (tid == 0) {
        for (uint i = 0; i < n_mel_entries; i++) {
            MelFBEntry e = mel_fb[i];
            si[e.mel] += sr[e.freq] * e.weight;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Log + write output
    if (tid < n_mels) {
        mel_out[tgid * n_mels + tid] = log(si[tid] + log_eps);
    }
}

// ---------------------------------------------------------------------------
// Per-channel mel normalize + transpose
//   mel_in:  [n_frames, 128] (float32)
//   mel_out: [128, n_valid]  (float32)
//   One threadgroup per mel channel (128 threadgroups).
// ---------------------------------------------------------------------------
kernel void mel_normalize_kernel(
    device const float* mel_in  [[buffer(0)]],
    device       float* mel_out [[buffer(1)]],
    constant     uint&  n_frames [[buffer(2)]],
    constant     uint&  n_valid  [[buffer(3)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    uint ch = tgid;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (uint i = tid; i < n_valid; i += tg_size) {
        sum += mel_in[i * 128 + ch];
    }
    sum = simd_sum(sum);
    threadgroup float s_buf[32];
    if (lane == 0) s_buf[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nw = tg_size / 32;
        sum = (lane < nw) ? s_buf[lane] : 0.0f;
        sum = simd_sum(sum);
    }
    threadgroup float s_mean;
    if (tid == 0) s_mean = sum / float(n_valid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (uint i = tid; i < n_valid; i += tg_size) {
        float diff = mel_in[i * 128 + ch] - mean;
        var_sum += diff * diff;
    }
    var_sum = simd_sum(var_sum);
    if (lane == 0) s_buf[simd_id] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint nw = tg_size / 32;
        var_sum = (lane < nw) ? s_buf[lane] : 0.0f;
        var_sum = simd_sum(var_sum);
    }
    threadgroup float s_inv_std;
    if (tid == 0) {
        // Bessel correction: divide by (n_valid - 1)
        float std_val = sqrt(var_sum / float(n_valid > 1 ? n_valid - 1 : 1));
        s_inv_std = 1.0f / (std_val + 1e-05f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = s_inv_std;

    // Pass 3: normalize + transpose [n_frames, 128] -> [128, n_valid]
    for (uint i = tid; i < n_valid; i += tg_size) {
        float val = (mel_in[i * 128 + ch] - mean) * inv_std;
        mel_out[ch * n_valid + i] = val;
    }
}
