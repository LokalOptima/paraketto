// conformer_base.h — buffers and decoder shared by the FP16 and FP8 CudaModels
//
// Defines:
//   CudaModelBase — GPU buffer pool + TDT decoder common to both variants
//
// The FP16 variant (conformer.h) and FP8 variant (conformer_fp8.h) each derive
// a CudaModel from this base, adding their own weight storage and encode_gpu().
// The pool carving and the decoder (LSTM + joint network, FP16 in both
// variants) are implemented once in conformer_base.cpp.

#pragma once

#include "model_defs.h"

namespace paraketto {

struct CudaModelBase {
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

    // --- Shared pool carving (implemented in conformer_base.cpp) ---
    // Bytes needed for the shared buffers above (aliased + persistent regions,
    // mel_fp32, argmax_out). Requires w and T_max to be set.
    size_t common_pool_bytes(int max_mel_frames) const;
    // Carve the shared buffers from `pool`. Returns the cursor just past them,
    // where variant-specific extras (e.g. FP16 qkv_w blocks) may be carved.
    char* carve_common_pool(char* pool, int max_mel_frames);

    // --- TDT decoder (identical in both variants, FP16 GEMMs via gemm.h) ---
    void  decoder_reset();
    half* decode_step(int enc_frame_idx, int prev_token);
    void  decoder_commit();
};

} // namespace paraketto
