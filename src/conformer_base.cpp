// conformer_base.cpp — CudaModelBase: shared pool carving + TDT decoder
//
// Compiled into all three binaries. The decoder uses the gemm.h interface,
// which every backend provides (cutlass_gemm.o or cublas_gemm.o at link time).

#include "conformer_base.h"
#include "common.h"
#include "kernels.h"
#include "gemm.h"

#include <algorithm>
#include <utility>

using namespace paraketto;

// ---------------------------------------------------------------------------
// Shared buffer pool
//
// Layout: [aliased region | persistent region | mel_fp32 | argmax | extras]
//
// Subsampling buffers (sub_buf, mel_fp16) are only used at the start of
// encode_gpu(). Conformer buffers (scores, ff_mid, etc.) are only used in the
// subsequent conformer loop. Since their lifetimes don't overlap, they share
// the same GPU memory region ("aliased region"):
//   aliased    = max(subsampling_phase, conformer_phase)
//   persistent = x, pos_enc, decoder/LSTM buffers, mel_fp32
// ---------------------------------------------------------------------------

namespace {

constexpr size_t ALIGN = 256;

// Aligned total bytes for an array of half-element counts
size_t region_bytes(const size_t* sizes, int n) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        off = (off + ALIGN - 1) & ~(ALIGN - 1);
        off += sizes[i] * sizeof(half);
    }
    return (off + ALIGN - 1) & ~(ALIGN - 1);
}

struct CommonSizes {
    size_t sub[3];       // subsampling-only buffers (dead after subsampling)
    size_t conf[16];     // conformer-only buffers (unused during subsampling)
    size_t persist[20];  // span both phases or used in decoder
    size_t mel_fp32_elems;
    size_t aliased_bytes;
    size_t persist_bytes;
};

CommonSizes compute_sizes(int max_mel_frames, int T_max, int d_output) {
    // Intermediate spatial dims for the subsampling conv chain:
    // conv.0: stride-2 on (max_mel_frames, 128) → (H2, W2)
    // conv.2: stride-2 on (H2, W2) → (H3, W3)
    int H2 = (max_mel_frames + 2 - 3) / 2 + 1;
    int W2 = (128             + 2 - 3) / 2 + 1;  // 64
    int H3 = (H2              + 2 - 3) / 2 + 1;
    int W3 = (W2              + 2 - 3) / 2 + 1;  // 32

    CommonSizes s;
    size_t sub[3] = {
        (size_t)(SUB_CHANNELS * H3 * W3),       // sub_buf[0]: peak at conv.2 output
        (size_t)(SUB_CHANNELS * H2 * W2),       // sub_buf[1]: peak at conv.0 output
        (size_t)(128 * max_mel_frames),         // mel_fp16
    };
    size_t conf[16] = {
        (size_t)(T_max * D_MODEL),              // ln_out
        (size_t)(T_max * D_FF),                 // ff_mid
        (size_t)(T_max * D_MODEL),              // ff_out
        (size_t)(T_max * 3 * D_MODEL),          // qkv
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // k
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // v
        (size_t)((2 * T_max) * D_MODEL),        // pos_proj
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // q_u
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // q_v_buf
        (size_t)(N_HEADS * T_max * T_max),      // scores
        (size_t)(N_HEADS * T_max * (2*T_max)),  // pos_scores
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // attn_out
        (size_t)(T_max * D_MODEL),              // mhsa_out
        (size_t)(T_max * D_CONV_PW),            // conv_mid
        (size_t)(T_max * D_MODEL),              // conv_glu
        (size_t)(T_max * D_MODEL),              // conv_dw
    };
    size_t persist[20] = {
        (size_t)(T_max * D_MODEL),              // x
        (size_t)((2 * T_max) * D_MODEL),        // pos_enc
        (size_t)(2 * D_PRED),                   // lstm_input
        (size_t)(4 * D_PRED),                   // lstm_gates
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_h[0], lstm_h[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_c[0], lstm_c[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_h_out[0], lstm_h_out[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_c_out[0], lstm_c_out[1]
        (size_t)(T_max * D_JOINT),              // enc_proj_all
        (size_t)D_JOINT, (size_t)D_JOINT,       // dec_proj_buf, joint_act
        (size_t)d_output,                       // joint_out
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[0]
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[1]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[0]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[1]
    };
    std::copy(sub, sub + 3, s.sub);
    std::copy(conf, conf + 16, s.conf);
    std::copy(persist, persist + 20, s.persist);
    s.mel_fp32_elems = (size_t)128 * max_mel_frames;
    s.aliased_bytes  = std::max(region_bytes(s.sub, 3), region_bytes(s.conf, 16));
    s.persist_bytes  = region_bytes(s.persist, 20);
    return s;
}

half* take(char*& cursor, size_t n) {
    cursor = (char*)(((uintptr_t)cursor + ALIGN - 1) & ~(ALIGN - 1));
    half* r = (half*)cursor;
    cursor += n * sizeof(half);
    return r;
}

} // namespace

size_t CudaModelBase::common_pool_bytes(int max_mel_frames) const {
    CommonSizes s = compute_sizes(max_mel_frames, T_max, w->config.d_output);
    return s.aliased_bytes + s.persist_bytes
         + ALIGN + s.mel_fp32_elems * sizeof(float)
         + ALIGN + 2 * sizeof(int);
}

char* CudaModelBase::carve_common_pool(char* pool, int max_mel_frames) {
    CommonSizes s = compute_sizes(max_mel_frames, T_max, w->config.d_output);

    // Aliased region: subsampling buffers and conformer scratch buffers share
    // the same base address. Safe because subsampling runs first, and once x
    // is written to the persistent region the subsampling buffers are never
    // touched again.
    char* sub_cur = pool;
    sub_buf[0] = take(sub_cur, s.sub[0]);
    sub_buf[1] = take(sub_cur, s.sub[1]);
    mel_fp16   = take(sub_cur, s.sub[2]);

    char* conf_cur = pool;  // same base address — intentionally aliased
    ln_out     = take(conf_cur, s.conf[0]);
    ff_mid     = take(conf_cur, s.conf[1]);
    ff_out     = take(conf_cur, s.conf[2]);
    qkv        = take(conf_cur, s.conf[3]);
    k          = take(conf_cur, s.conf[4]);
    v          = take(conf_cur, s.conf[5]);
    pos_proj   = take(conf_cur, s.conf[6]);
    q_u        = take(conf_cur, s.conf[7]);
    q_v_buf    = take(conf_cur, s.conf[8]);
    scores     = take(conf_cur, s.conf[9]);
    pos_scores = take(conf_cur, s.conf[10]);
    attn_out   = take(conf_cur, s.conf[11]);
    mhsa_out   = take(conf_cur, s.conf[12]);
    conv_mid   = take(conf_cur, s.conf[13]);
    conv_glu   = take(conf_cur, s.conf[14]);
    conv_dw    = take(conf_cur, s.conf[15]);

    // Persistent region (after aliased region)
    char* pers_cur = pool + s.aliased_bytes;
    x              = take(pers_cur, s.persist[0]);
    pos_enc        = take(pers_cur, s.persist[1]);
    lstm_input     = take(pers_cur, s.persist[2]);
    lstm_gates     = take(pers_cur, s.persist[3]);
    lstm_h[0]      = take(pers_cur, s.persist[4]);
    lstm_h[1]      = take(pers_cur, s.persist[5]);
    lstm_c[0]      = take(pers_cur, s.persist[6]);
    lstm_c[1]      = take(pers_cur, s.persist[7]);
    lstm_h_out[0]  = take(pers_cur, s.persist[8]);
    lstm_h_out[1]  = take(pers_cur, s.persist[9]);
    lstm_c_out[0]  = take(pers_cur, s.persist[10]);
    lstm_c_out[1]  = take(pers_cur, s.persist[11]);
    enc_proj_all   = take(pers_cur, s.persist[12]);
    dec_proj_buf   = take(pers_cur, s.persist[13]);
    joint_act      = take(pers_cur, s.persist[14]);
    joint_out      = take(pers_cur, s.persist[15]);
    lstm_combined_w[0]    = take(pers_cur, s.persist[16]);
    lstm_combined_w[1]    = take(pers_cur, s.persist[17]);
    lstm_combined_bias[0] = take(pers_cur, s.persist[18]);
    lstm_combined_bias[1] = take(pers_cur, s.persist[19]);

    // mel_fp32 (float, not half) — persistent
    pers_cur = (char*)(((uintptr_t)pers_cur + ALIGN - 1) & ~(ALIGN - 1));
    mel_fp32 = (float*)pers_cur;
    pers_cur += s.mel_fp32_elems * sizeof(float);

    // argmax_out (int[2]) — persistent
    pers_cur = (char*)(((uintptr_t)pers_cur + ALIGN - 1) & ~(ALIGN - 1));
    argmax_out = (int*)pers_cur;
    pers_cur += 2 * sizeof(int);

    return pers_cur;
}

// ---------------------------------------------------------------------------
// TDT decoder — LSTM prediction network + joint network (FP16 in all variants)
// ---------------------------------------------------------------------------

void CudaModelBase::decoder_reset() {
    CUDA_CHECK(cudaMemsetAsync(lstm_h[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_h[1], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[1], 0, D_PRED * sizeof(half), stream));
}

half* CudaModelBase::decode_step(int enc_frame_idx, int prev_token) {
    auto gnn_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nn_bias(stream, X, m, k, W, n, bias, Y);
    };
    auto gnt_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nt_bias(stream, X, m, k, W, n, bias, Y);
    };

    // 1. Embed + concat with h[0] for LSTM0 input: lstm_input = [embed; h[0]]
    embed_concat_fp16(w->embed_w, prev_token, lstm_h[0], lstm_input, D_PRED, stream);

    // 2. LSTM layer 0 — single GEMM with combined [W_ih|W_hh] weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[0], 4 * D_PRED, lstm_combined_bias[0], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[0], lstm_h_out[0], lstm_c_out[0], D_PRED, stream);

    // 3. Concat h_out[0] with h[1] for LSTM1 input: lstm_input = [h_out[0]; h[1]]
    concat_vectors_fp16(lstm_h_out[0], lstm_h[1], lstm_input, D_PRED, stream);

    // 4. LSTM layer 1 — single GEMM with combined weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[1], 4 * D_PRED, lstm_combined_bias[1], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[1], lstm_h_out[1], lstm_c_out[1], D_PRED, stream);

    // 5. Joint network — GEMMs for dec_proj + out_proj (multi-SM beats single-SM for 2MB weights)
    half* enc_proj_t = enc_proj_all + enc_frame_idx * D_JOINT;
    gnn_b(lstm_h_out[1], 1, D_PRED, w->dec_proj_w, D_JOINT, w->dec_proj_b, dec_proj_buf);
    add_relu_fp16(enc_proj_t, dec_proj_buf, joint_act, D_JOINT, stream);
    gnn_b(joint_act, 1, D_JOINT, w->out_proj_w, w->config.d_output, w->out_proj_b, joint_out);

    return joint_out;
}

// Swap LSTM state after non-blank token
void CudaModelBase::decoder_commit() {
    std::swap(lstm_h[0], lstm_h_out[0]);
    std::swap(lstm_c[0], lstm_c_out[0]);
    std::swap(lstm_h[1], lstm_h_out[1]);
    std::swap(lstm_c[1], lstm_c_out[1]);
}
