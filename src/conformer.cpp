// conformer.cpp — FP16 CudaModel inference (CUTLASS or cuBLAS backend)
//
// Uses the gemm.h unified interface — build with -DUSE_CUBLAS for cuBLAS,
// default is CUTLASS (no cuBLAS dependency).

#include "conformer.h"
#include "common.h"
#include "kernels.h"
#include "gemm.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace paraketto;

void CudaModel::init(const Weights& weights, cudaStream_t s, int max_mel_frames) {
    w = &weights;
    stream = s;
    T_max = max_mel_frames / 8 + 10;  // encoder frames after 8x downsampling

    gemm_init(stream);

    // --- Pooled GPU allocation ---
    // Shared buffers (aliased + persistent regions, mel_fp32, argmax_out) are
    // carved by CudaModelBase; this variant appends the pre-concatenated FP16
    // QKV weight blocks after them.
    constexpr size_t ALIGN = 256;
    size_t qkv_w_elem = (size_t)D_MODEL * 3 * D_MODEL;
    size_t qkv_w_block = (qkv_w_elem * sizeof(half) + ALIGN - 1) & ~(ALIGN - 1);
    size_t qkv_w_bytes = (size_t)N_BLOCKS * qkv_w_block;

    size_t pool_bytes = common_pool_bytes(max_mel_frames) + ALIGN + qkv_w_bytes;

    char* pool;
    CUDA_CHECK(cudaMalloc(&pool, pool_bytes));
    gpu_pool = pool;

    char* extra_cur = carve_common_pool(pool, max_mel_frames);

    // QKV weight blocks (persistent, used in conformer loop)
    auto take = [](char*& cursor, size_t n) -> half* {
        cursor = (char*)(((uintptr_t)cursor + ALIGN - 1) & ~(ALIGN - 1));
        half* r = (half*)cursor;
        cursor += n * sizeof(half);
        return r;
    };
    for (int b = 0; b < N_BLOCKS; b++)
        qkv_w[b] = take(extra_cur, qkv_w_elem);

    // Pre-concatenate QKV weights using cudaMemcpy2D (72 calls vs 73K row-by-row)
    for (int b = 0; b < N_BLOCKS; b++) {
        size_t dst_pitch = 3 * D_MODEL * sizeof(half);
        size_t src_pitch = D_MODEL * sizeof(half);
        size_t width = D_MODEL * sizeof(half);
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b], dst_pitch,
            weights.blocks[b].q_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b] + D_MODEL, dst_pitch,
            weights.blocks[b].k_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b] + 2 * D_MODEL, dst_pitch,
            weights.blocks[b].v_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
    }

    // Pre-combine LSTM weights: W_combined[4*D, 2*D] = [W_ih | W_hh] per layer
    // Same cudaMemcpy2D pattern as QKV concatenation above.
    {
        const half* w_ih[2] = { weights.lstm0_w_ih, weights.lstm1_w_ih };
        const half* w_hh[2] = { weights.lstm0_w_hh, weights.lstm1_w_hh };
        const half* bias[2] = { weights.lstm0_bias, weights.lstm1_bias };
        size_t dst_pitch = 2 * D_PRED * sizeof(half);
        size_t src_pitch = D_PRED * sizeof(half);
        size_t width = D_PRED * sizeof(half);
        for (int layer = 0; layer < 2; layer++) {
            // Left half: W_ih
            CUDA_CHECK(cudaMemcpy2DAsync(
                lstm_combined_w[layer], dst_pitch,
                w_ih[layer], src_pitch,
                width, 4 * D_PRED, cudaMemcpyDeviceToDevice, stream));
            // Right half: W_hh
            CUDA_CHECK(cudaMemcpy2DAsync(
                lstm_combined_w[layer] + D_PRED, dst_pitch,
                w_hh[layer], src_pitch,
                width, 4 * D_PRED, cudaMemcpyDeviceToDevice, stream));
            // Pre-add biases: combined_bias = b_ih + b_hh
            residual_add_fp16(bias[layer], bias[layer] + 4 * D_PRED,
                              lstm_combined_bias[layer], 4 * D_PRED, 1.0f, stream);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pre-compute position encoding for T_max (reused via offset for any T <= T_max)
    generate_pos_encoding_gpu(pos_enc, T_max, D_MODEL, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// CudaModel::free
// ---------------------------------------------------------------------------

void CudaModel::free() {
    gemm_free();
    // All inference buffers are carved from a single pooled allocation
    if (gpu_pool) { cudaFree(gpu_pool); gpu_pool = nullptr; }
}

// GEMM calls use gemm.h interface directly — backend selected at link time.

// Position encoding generation moved to GPU kernel (generate_pos_encoding_gpu)

int CudaModel::encode_gpu(int T_mel) {
    // Local GEMM helpers that capture stream
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nn(stream, X, m, k, W, n, Y);
    };
    auto gnt = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nt(stream, X, m, k, W, n, Y);
    };
    auto gnn_bias = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nn_bias(stream, X, m, k, W, n, bias, Y);
    };

    // 1. Cast mel to FP16, then transpose
    //    mel_fp32 is [128, T_mel] on GPU
    //    ONNX model transposes to [T_mel, 128] before conv2d
    cast_fp32_to_fp16(mel_fp32, mel_fp16, 128 * T_mel, stream);

    // Transpose mel from [128, T_mel] to [T_mel, 128]
    transpose_fp16(mel_fp16, sub_buf[0], 128, T_mel, stream);
    // sub_buf[0] is now [T_mel, 128] which acts as [C_in=1, H=T_mel, W=128] for conv2d

    // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
    //    conv.0 (1→256, 3x3, s=2) → bias+ReLU
    int H = T_mel, W = 128;
    int H2 = (H + 2 * 1 - 3) / 2 + 1;
    int W2 = (W + 2 * 1 - 3) / 2 + 1;  // 64
    conv2d_fp16(sub_buf[0], w->sub_conv[0].weight, nullptr,
                sub_buf[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1, stream);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[0].bias, SUB_CHANNELS, H2 * W2, stream);
    H = H2; W = W2;

    //    conv.2 (depthwise 256, 3x3, s=2) → conv.3 (pointwise 256→256, 1x1) → bias+ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[2].weight, w->sub_conv[2].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;   // 32
    // Pointwise 1x1 conv = GEMM: out[256, H*W] = weight[256, 256] @ in[256, H*W]
    gnn(/*X=*/w->sub_conv[3].weight, SUB_CHANNELS, SUB_CHANNELS,
        /*W=*/sub_buf[0], H * W, /*Y=*/sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[3].bias, SUB_CHANNELS, H * W, stream);

    //    conv.5 (depthwise 256, 3x3, s=2) → conv.6 (pointwise 256→256, 1x1) → bias+ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[5].weight, w->sub_conv[5].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;   // T' (encoder frames)
    W = (W + 2 * 1 - 3) / 2 + 1;   // 16
    // Pointwise 1x1 conv = GEMM: out[256, H*W] = weight[256, 256] @ in[256, H*W]
    gnn(/*X=*/w->sub_conv[6].weight, SUB_CHANNELS, SUB_CHANNELS,
        /*W=*/sub_buf[0], H * W, /*Y=*/sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[6].bias, SUB_CHANNELS, H * W, stream);

    // Reshape [256, H, 16] → [H, 256*16] = [T', 4096] then linear → [T', 1024]
    // This is [C, H, W] → [H, C*W] (permute(1,0,2) + flatten)
    int T = H;  // encoder frames = time dim after 3x stride-2
    reshape_chw_to_hcw_fp16(sub_buf[1], sub_buf[0], SUB_CHANNELS, T, W, stream);
    // sub_buf[0] is now [T, 4096]
    if (w->sub_out_b)
        gnn_bias(sub_buf[0], T, SUB_CHANNELS * W, w->sub_out_w, D_MODEL, w->sub_out_b, x);
    else
        gnn(sub_buf[0], T, SUB_CHANNELS * W, w->sub_out_w, D_MODEL, x);

    // 3. Position encoding: slice from pre-computed T_max encoding
    //    pos_enc for T is at offset (T_max - T) * D_MODEL in the T_max encoding
    half* pos_enc_T = pos_enc + (T_max - T) * D_MODEL;
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = w->blocks[blk];

        // --- FF1 (half-step residual) ---
        layer_norm_fp16(x, b.ff1_ln_w, b.ff1_ln_b, ln_out, T, D_MODEL, NORM_EPS, stream);

        // FF1 linear1 + SiLU: [T,1024] × [1024,4096] → [T,4096]
        gnn(ln_out, T, D_MODEL, b.ff1_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn(ff_mid, T, D_FF, b.ff1_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.mhsa_ln_w, b.mhsa_ln_b, ln_out, T, D_MODEL, NORM_EPS, stream);

        {
            // Fused QKV projection: [T, D] × [D, 3D] → [T, 3D]
            gnn(ln_out, T, D_MODEL, qkv_w[blk], 3 * D_MODEL, qkv);

            // Fused split+transpose+bias: [T, 3D] → q_u, q_v (with bias), K, V
            half* K_h = k;
            half* V_h = v;
            split_transpose_qkv_bias_fp16(qkv, b.pos_bias_u, b.pos_bias_v,
                                           q_u, q_v_buf, K_h, V_h,
                                           T, N_HEADS, HEAD_DIM, stream);

            // Position encoding projection
            half* pos_temp = pos_proj;
            gnn(pos_enc_T, pos_len, D_MODEL, b.pos_w, D_MODEL, pos_temp);

            // Position scores via strided batched GEMM
            // Row-major NT: pos_scores[h,T,pos_len] = q_v[h,T,HEAD_DIM] @ pos[h,pos_len,HEAD_DIM]^T
            // pos_temp has non-standard ld=D_MODEL (interleaved heads), stride=HEAD_DIM between heads
            batched_gemm_nt_ex(stream,
                q_v_buf, HEAD_DIM, (long long)T * HEAD_DIM,        // A: [T, HEAD_DIM] per head
                pos_temp, D_MODEL, (long long)HEAD_DIM,            // B: [pos_len, HEAD_DIM] ld=D_MODEL
                pos_scores, pos_len, (long long)T * pos_len,       // C: [T, pos_len] per head
                N_HEADS, T, pos_len, HEAD_DIM);

            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            // Content attention scores: [8, T, T] = q_u @ K^T
            batched_gemm_nt(stream, q_u, K_h, scores,
                            N_HEADS, T, T, HEAD_DIM,
                            (long long)T * HEAD_DIM, (long long)T * HEAD_DIM, (long long)T * T);
            // Fused: (content + pos_skew) * scale + softmax
            fused_score_softmax_fp16(scores, pos_scores, scores,
                                      N_HEADS, T, scale, stream);
            // Weighted sum: [8, T, 128] = scores @ V
            batched_gemm_nn(stream, scores, V_h, attn_out,
                            N_HEADS, T, HEAD_DIM, T,
                            (long long)T * T, (long long)T * HEAD_DIM, (long long)T * HEAD_DIM);
            // Transpose [8, T, 128] → [T, 8, 128] = [T, 1024]
            transpose_0213_fp16(attn_out, ff_out, N_HEADS, T, HEAD_DIM, stream);

            // Output projection
            gnn(ff_out, T, D_MODEL, b.out_w, D_MODEL, mhsa_out);
        }

        // Fused: x += mhsa_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.conv_ln_w, b.conv_ln_b, ln_out, T, D_MODEL, NORM_EPS, stream);

        // Pointwise conv1 + GLU
        gnt(ln_out, T, D_MODEL, b.conv_pw1_w, D_CONV_PW, conv_mid);
        glu_fp16(conv_mid, conv_glu, T, D_MODEL, stream);

        // Fused depthwise conv 1D k=9 + SiLU
        depthwise_conv1d_k9_silu_fp16(conv_glu, b.conv_dw_w, b.conv_dw_b,
                                       conv_dw, T, D_MODEL, stream);

        // Pointwise conv2
        gnt(conv_dw, T, D_MODEL, b.conv_pw2_w, D_MODEL, mhsa_out);

        // Fused: x += conv_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.ff2_ln_w, b.ff2_ln_b, ln_out, T, D_MODEL, NORM_EPS, stream);

        // FF2 linear1 + SiLU
        gnn(ln_out, T, D_MODEL, b.ff2_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn(ff_mid, T, D_FF, b.ff2_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.final_ln_w, b.final_ln_b, x, T, D_MODEL, NORM_EPS, stream);
    }

    // x now holds encoder output [T, D_MODEL] in FP16

    // Precompute encoder projections for all T frames (used by fused joint kernel)
    gnn_bias(x, T, D_MODEL, w->enc_proj_w, D_JOINT, w->enc_proj_b, enc_proj_all);

    return T;
}

// Decoder (decoder_reset / decode_step / decoder_commit) is shared with the
// FP8 variant — see CudaModelBase in conformer_base.cpp.
