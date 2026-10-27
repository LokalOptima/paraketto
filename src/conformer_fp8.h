// conformer_fp8.h — FP8 CudaModel (CUTLASS E4M3 backend)
//
// Selected by building with -DPARAKETTO_FP8=1: paraketto.h includes this
// header instead of conformer.h. Shared buffers and the TDT decoder live in
// CudaModelBase (conformer_base.h). This variant adds FP8 quantization
// fields: E4M3 weight copies, per-tensor scales, and calibrated activation scales.
//
// This file reuses the CONFORMER_H_ include guard so that a stray direct
// #include "conformer.h" in the same TU becomes a no-op — the FP8 CudaModel
// definition must be the only one in an FP8 build.

#ifndef CONFORMER_H_
#define CONFORMER_H_
#ifndef PARAKETTO_FP8
#define PARAKETTO_FP8 1
#endif

#include "conformer_base.h"

namespace paraketto {

struct CudaModel : CudaModelBase {
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
    float*   fp8_act_site_scales      = nullptr;  // [N_FP8_ACT_SITES] embedded or baked defaults
    float*   fp8_alpha_products       = nullptr;  // [N_FP8_ACT_SITES] pre-computed w_scale * act_scale
    uint8_t* fp8_act_buf              = nullptr;
    int*     fp8_calib_amax_buf       = nullptr;  // dynamic calibration scratch
    bool     fp8_embedded_scales      = false;

    /// fp8_path: path to offline-quantized PRKTFP8 format-2/3 weights.
    /// fp8_prefetch: pre-populated mmap of paraketto-fp8.bin (from background prefetch thread).
    void init(const Weights& weights, cudaStream_t s, int max_mel_frames,
              const char* fp8_path = nullptr,
              const void* fp8_prefetch = nullptr, size_t fp8_prefetch_size = 0);
    void free();

    int encode_gpu(int T_mel);
};

} // namespace paraketto

#endif  // CONFORMER_H_
