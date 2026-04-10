// conformer_fp8.h — FP8 CudaModel (cublasLt E4M3 backend)
//
// Included via -include flag for the FP8 build, overriding conformer.h.
// Defines PARAKETTO_FP8 and extends CudaModel with FP8 quantization fields.
//
// IMPORTANT: This file intentionally reuses the CONFORMER_H_ include guard so
// that when the compiler processes it first (via -include), the subsequent
// #include "conformer.h" in paraketto_cuda.cpp becomes a no-op. This lets the
// FP8 CudaModel definition cleanly replace the FP16 one at compile time.

#ifndef CONFORMER_H_
#define CONFORMER_H_
#define PARAKETTO_FP8 1

#include "model_defs.h"

namespace paraketto {

// ---------------------------------------------------------------------------
// CudaModel — FP8 variant with extra quantization fields
// ---------------------------------------------------------------------------

struct CudaModel {
    cudaStream_t   stream = nullptr;
    const Weights* w      = nullptr;

    int T_max = 0;

    half* qkv_w[N_BLOCKS];
    half* lstm_combined_w[2];
    half* lstm_combined_bias[2];
    half* lstm_input;

    void* gpu_pool = nullptr;

    float* mel_fp32    = nullptr;
    half*  mel_fp16    = nullptr;
    half*  sub_buf[2];
    half*  x           = nullptr;
    half*  ln_out      = nullptr;
    half*  ff_mid      = nullptr;
    half*  ff_out      = nullptr;
    half*  qkv         = nullptr;
    half*  k           = nullptr;
    half*  v           = nullptr;
    half*  pos_enc     = nullptr;
    half*  pos_proj    = nullptr;
    half*  q_u         = nullptr;
    half*  q_v_buf     = nullptr;
    half*  scores      = nullptr;
    half*  pos_scores  = nullptr;
    half*  attn_out    = nullptr;
    half*  mhsa_out    = nullptr;
    half*  conv_mid    = nullptr;
    half*  conv_glu    = nullptr;
    half*  conv_dw     = nullptr;

    half* lstm_gates    = nullptr;
    half* lstm_h[2];
    half* lstm_c[2];
    half* lstm_h_out[2];
    half* lstm_c_out[2];
    half* enc_proj_all  = nullptr;
    half* dec_proj_buf  = nullptr;
    half* joint_act     = nullptr;
    half* joint_out     = nullptr;
    int*  argmax_out    = nullptr;

    // FP8 quantization data
    static constexpr int N_FP8_SCALES    = N_BLOCKS * 9 + 6;   // 222
    static constexpr int N_FP8_ACT_SITES = N_BLOCKS * 9 + 2;   // 218

    void*    fp8_pool                 = nullptr;
    uint8_t* fp8_qkv_w[N_BLOCKS]     = {};
    uint8_t* fp8_ff1_w1[N_BLOCKS]    = {};
    uint8_t* fp8_ff1_w2[N_BLOCKS]    = {};
    uint8_t* fp8_ff2_w1[N_BLOCKS]    = {};
    uint8_t* fp8_ff2_w2[N_BLOCKS]    = {};
    uint8_t* fp8_pos_w[N_BLOCKS]     = {};
    uint8_t* fp8_out_w[N_BLOCKS]     = {};
    uint8_t* fp8_conv_pw1_w[N_BLOCKS] = {};
    uint8_t* fp8_conv_pw2_w[N_BLOCKS] = {};
    uint8_t* fp8_sub_out_w            = nullptr;
    uint8_t* fp8_enc_proj_w           = nullptr;
    uint8_t* fp8_lstm_combined_w[2]   = {};
    uint8_t* fp8_dec_proj_w           = nullptr;
    uint8_t* fp8_out_proj_w           = nullptr;
    float*   fp8_scales               = nullptr;  // [N_FP8_SCALES]
    float*   fp8_act_site_scales      = nullptr;  // [N_FP8_ACT_SITES]
    float*   fp8_alpha_products       = nullptr;  // [N_FP8_ACT_SITES] pre-computed w_scale * act_scale
    uint8_t* fp8_act_buf              = nullptr;
    int*     fp8_amax_buf             = nullptr;
    bool     fp8_calibrated           = false;

    /// fp8_path: path to paraketto-fp8.bin (load if exists, save after quantization).
    /// fp8_prefetch: pre-populated mmap of paraketto-fp8.bin (from background prefetch thread).
    void init(const Weights& weights, cudaStream_t s, int max_mel_frames,
              const char* fp8_path = nullptr,
              const void* fp8_prefetch = nullptr, size_t fp8_prefetch_size = 0);
    void free();

    int   encode_gpu(int T_mel);
    void  decoder_reset();
    half* decode_step(int enc_frame_idx, int prev_token);
    void  decoder_commit();
};

} // namespace paraketto

#endif  // CONFORMER_H_
