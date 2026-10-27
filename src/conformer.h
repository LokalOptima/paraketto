// conformer.h — FP16 encoder, decoder, and GPU buffers
// GEMM backend (CUTLASS or cuBLAS) is selected at link time via gemm.h.

#pragma once

#include "model_defs.h"

namespace paraketto {

struct CudaModel {
    cudaStream_t   stream = nullptr;
    const Weights* w      = nullptr;

    int T_max = 0;  // max encoder frames (after 8x downsampling)

    // Pre-combined LSTM weights [4*D_PRED, 2*D_PRED] and biases [4*D_PRED]
    half* lstm_combined_w[2];
    half* lstm_combined_bias[2];
    half* lstm_input;  // [2*D_PRED] runtime concat buffer

    // Single pooled GPU allocation for all inference buffers
    void* gpu_pool = nullptr;

    // Encoder buffers
    float* mel_fp32    = nullptr;  // [128, T_mel]
    half*  mel_fp16    = nullptr;  // [128, T_mel]
    half*  sub_buf[2];             // subsampling ping-pong
    half*  x           = nullptr;  // [T', D_MODEL] main activation
    half*  ln_out      = nullptr;
    half*  ff_mid      = nullptr;  // [T', D_FF]
    half*  ff_out      = nullptr;
    half*  qkv         = nullptr;  // [T', 3*D_MODEL]
    half*  k           = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half*  v           = nullptr;
    half*  pos_enc     = nullptr;  // [2*T_max-1, D_MODEL]
    half*  pos_proj    = nullptr;
    half*  q_u         = nullptr;
    half*  q_v_buf     = nullptr;
    half*  scores      = nullptr;  // [N_HEADS, T', T']
    half*  pos_scores  = nullptr;  // [N_HEADS, T', 2*T'-1]
    half*  attn_out    = nullptr;
    half*  mhsa_out    = nullptr;
    half*  conv_mid    = nullptr;  // [T', D_CONV_PW]
    half*  conv_glu    = nullptr;
    half*  conv_dw     = nullptr;

    // Decoder buffers
    half* lstm_gates    = nullptr;  // [4*D_PRED]
    half* lstm_h[2];                // [D_PRED] per layer
    half* lstm_c[2];
    half* lstm_h_out[2];
    half* lstm_c_out[2];
    half* enc_proj_all  = nullptr;  // [T_max, D_JOINT]
    half* dec_proj_buf  = nullptr;  // [D_JOINT]
    half* joint_act     = nullptr;
    half* joint_out     = nullptr;  // [D_OUTPUT]
    int*  argmax_out    = nullptr;  // [2]: token, step

    // Pre-concatenated QKV weights per block [D_MODEL, 3*D_MODEL]
    half* qkv_w[N_BLOCKS];

    void init(const Weights& weights, cudaStream_t s, int max_mel_frames);
    void free();
    int encode_gpu(int T_mel);

    // TDT decoder (LSTM prediction network + joint network)
    void  decoder_reset();
    half* decode_step(int enc_frame_idx, int prev_token);
    void  decoder_commit();

private:
    void allocate_buffers(int max_mel_frames);
};

} // namespace paraketto
