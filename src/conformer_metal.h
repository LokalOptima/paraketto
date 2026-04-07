// conformer_metal.h — Weight loading + Metal inference for Parakeet conformer
//
// Defines:
//   MetalModel — pre-allocated buffers + GEMM backend for forward passes
//
// Model constants, ModelConfig, and Weights layout shared with the CUDA path
// via model_defs.h (with CUDA includes guarded behind #ifdef __CUDACC__).

#ifndef CONFORMER_METAL_H_
#define CONFORMER_METAL_H_

#include "metal_context.h"
#include <cstddef>
#include <cstdint>

// Model constants (duplicated from model_defs.h to avoid CUDA dependency)
static constexpr int D_MODEL    = 1024;
static constexpr int D_FF       = 4096;
static constexpr int N_HEADS    = 8;
static constexpr int HEAD_DIM   = D_MODEL / N_HEADS;  // 128
static constexpr int N_BLOCKS   = 24;
static constexpr int D_CONV_PW  = 2048;
static constexpr int CONV_K     = 9;
static constexpr int SUB_CHANNELS = 256;
static constexpr int D_PRED     = 640;
static constexpr int D_JOINT    = 640;

struct MetalModel {
    MetalContext ctx;

    int T_max = 0;  // max encoder frames (after 8x downsampling)

    // Runtime model config
    int n_vocab  = 1025;
    int d_output = 1030;
    int blank_id = 1024;

    // GPU memory: single pooled MTLBuffer for all inference buffers
    // All "pointers" are byte offsets into this buffer.
    void* gpu_pool_handle = nullptr;
    size_t gpu_pool_bytes = 0;

    // Weight buffer: MTLBuffer holding the model weights
    void* weight_handle = nullptr;
    size_t weight_bytes = 0;

    // --- Buffer offsets (bytes) into gpu_pool ---

    // Encoder (subsampling phase — aliased with conformer phase)
    size_t mel_fp32_off = 0;
    size_t mel_fp16_off = 0;
    size_t sub_buf_off[2] = {};

    // Encoder (conformer phase — aliased with subsampling)
    size_t x_off = 0;
    size_t ln_out_off = 0;
    size_t ff_mid_off = 0;
    size_t ff_out_off = 0;
    size_t qkv_off = 0;
    size_t q_off = 0, k_off = 0, v_off = 0;
    size_t pos_enc_off = 0;
    size_t pos_proj_off = 0;
    size_t q_u_off = 0, q_v_buf_off = 0;
    size_t scores_off = 0, pos_scores_off = 0;
    size_t attn_out_off = 0, mhsa_out_off = 0;
    size_t conv_mid_off = 0, conv_glu_off = 0, conv_dw_off = 0;

    // Decoder
    size_t lstm_input_off = 0;
    size_t lstm_gates_off = 0;
    size_t lstm_h_off[2] = {}, lstm_c_off[2] = {};
    size_t lstm_h_out_off[2] = {}, lstm_c_out_off[2] = {};
    size_t enc_proj_all_off = 0;
    size_t dec_proj_buf_off = 0, joint_act_off = 0;
    size_t joint_out_off = 0;
    size_t argmax_out_off = 0;

    // Pre-concatenated QKV weights per block
    size_t qkv_w_off[N_BLOCKS] = {};

    // Pre-combined LSTM weights and biases
    size_t lstm_combined_w_off[2] = {};
    size_t lstm_combined_bias_off[2] = {};

    // --- Weight offsets (bytes) into weight buffer ---
    // These mirror the Weights struct layout from model_defs.h

    // Lifecycle
    void init(const char* weights_path, int max_mel_frames);
    void free();

    // Inference
    int  encode_gpu(int T_mel);
    void decoder_reset();
    int  decode_step(int enc_frame_idx, int prev_token);  // returns token
    void decoder_commit();
};

#endif  // CONFORMER_METAL_H_
