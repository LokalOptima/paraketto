// conformer_fp8.cpp — FP8 CudaModel inference (CUTLASS FP8 backend)
//
// FP8 E4M3 weight quantization at init time, FP8 GEMMs via CUTLASS 2.x.
// FP16 sub-conv and batched attention GEMMs via CUTLASS FP16.
// No cuBLAS or cublasLt dependency.

#include "conformer_fp8.h"
#include "common.h"
#include "kernels.h"
#include "kernels_fp8.h"
#include "cutlass_gemm_fp8.h"
#include "cutlass_gemm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>

using namespace paraketto;

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Weight loading is in weights.cpp (shared with FP16 backends).

// Baked FP8 activation scales from reference utterance (1089-134686-0000.wav).
// This specific utterance's absmax values produce a quantization grid that
// works well across all datasets including difficult/noisy audio.  Validated
// by sweeping 21 single-utterance calibrations: this one is a consistent
// outlier at 16.5% difficult WER vs the 20% mean.
static const float FP8_BAKED_ACT_SCALES[218] = {
    2.53295898e-03f, 1.29255027e-01f, 6.50133414e-04f, 2.23214296e-03f, 2.97328411e-03f, 1.51192797e-02f,  // blk 0
    8.53097066e-02f, 3.02211214e-02f, 1.04282923e-01f, 2.27050781e-02f, 9.35581792e-03f, 1.66429789e-03f,
    2.23214296e-03f, 4.23976360e-03f, 8.44900962e-03f, 2.92271208e-02f, 3.93676758e-03f, 1.57557894e-02f,  // blk 1
    2.05601286e-02f, 1.19716097e-02f, 1.43541605e-03f, 2.23214296e-03f, 3.70134623e-03f, 4.72150510e-03f,  // blk 2
    7.94328935e-03f, 3.80597799e-03f, 1.26517164e-02f, 2.10832860e-02f, 2.00544093e-02f, 1.49100169e-03f,
    2.23214296e-03f, 4.05229861e-03f, 3.08009563e-03f, 8.54928140e-03f, 7.80378049e-03f, 7.91713130e-03f,  // blk 3
    1.93394255e-02f, 1.52587891e-02f, 1.15966797e-03f, 2.23214296e-03f, 3.37001262e-03f, 3.67736816e-03f,  // blk 4
    8.89369380e-03f, 8.49696528e-03f, 8.82393960e-03f, 2.72565577e-02f, 9.29478277e-03f, 9.10077768e-04f,
    2.23214296e-03f, 3.57055664e-03f, 2.66810833e-03f, 7.54220132e-03f, 6.04683999e-03f, 5.34057617e-03f,  // blk 5
    2.08914615e-02f, 1.28348218e-02f, 6.59397687e-04f, 2.23214296e-03f, 2.68990663e-03f, 2.06320616e-03f,  // blk 6
    8.07407964e-03f, 3.67300841e-03f, 7.34165730e-03f, 1.95835661e-02f, 1.19280135e-02f, 5.25883283e-04f,
    2.23214296e-03f, 2.90352968e-03f, 2.13950011e-03f, 8.82393960e-03f, 3.59235494e-03f, 1.11781526e-02f,  // blk 7
    1.87116358e-02f, 1.56250000e-02f, 8.61576642e-04f, 2.23214296e-03f, 2.71824421e-03f, 3.28063965e-03f,  // blk 8
    5.89861209e-03f, 3.59889446e-03f, 1.81012843e-02f, 1.04893278e-02f, 1.44653320e-02f, 8.86644644e-04f,
    2.23214296e-03f, 2.37819133e-03f, 2.72696349e-03f, 1.03498185e-02f, 2.09699362e-03f, 9.11167730e-03f,  // blk 9
    5.71550662e-03f, 1.34102954e-02f, 5.12259372e-04f, 2.23214296e-03f, 3.56183737e-03f, 3.50734172e-03f,  // blk 10
    1.26691544e-02f, 2.08500447e-03f, 1.27214706e-02f, 5.58907632e-03f, 1.40642440e-02f, 8.37053580e-04f,
    2.23214296e-03f, 2.55911681e-03f, 3.80815775e-03f, 1.62004735e-02f, 2.25394103e-03f, 1.24511719e-02f,  // blk 11
    4.37709270e-03f, 1.28173828e-02f, 8.59396823e-04f, 2.23214296e-03f, 1.74386159e-03f, 3.48772318e-03f,  // blk 12
    4.43812786e-03f, 2.15039938e-03f, 1.17623461e-02f, 6.02940144e-03f, 1.12566268e-02f, 7.36236572e-04f,
    2.23214296e-03f, 2.29099812e-03f, 3.52696003e-03f, 4.03050007e-03f, 1.45939423e-03f, 9.15527344e-03f,  // blk 13
    3.49644246e-03f, 6.36073528e-03f, 6.80650992e-04f, 2.23214296e-03f, 2.41742819e-03f, 3.28935892e-03f,  // blk 14
    6.88389363e-03f, 1.56402588e-03f, 1.12827849e-02f, 5.42340940e-03f, 1.13438200e-02f, 5.41414542e-04f,
    2.23214296e-03f, 1.69154571e-03f, 3.40270996e-03f, 4.76946170e-03f, 2.57001608e-03f, 1.33579802e-02f,  // blk 15
    6.97108684e-03f, 8.64519365e-03f, 6.32694806e-04f, 2.23214296e-03f, 3.44412657e-03f, 2.35421327e-03f,  // blk 16
    1.20413648e-02f, 1.91497803e-03f, 8.24846514e-03f, 7.58579792e-03f, 1.09078540e-02f, 6.96454721e-04f,
    2.23214296e-03f, 2.62233196e-03f, 2.72260397e-03f, 5.94656821e-03f, 3.00380168e-03f, 8.00868403e-03f,  // blk 17
    4.94384766e-03f, 9.10295732e-03f, 5.60760498e-04f, 2.23214296e-03f, 2.24304199e-03f, 3.00162169e-03f,  // blk 18
    8.13947432e-03f, 4.83921589e-03f, 1.02975024e-02f, 3.28063965e-03f, 1.01754321e-02f, 6.39234262e-04f,
    2.23214296e-03f, 2.40870891e-03f, 3.01034120e-03f, 9.29478277e-03f, 4.91768960e-03f, 1.37677873e-02f,  // blk 19
    6.59615640e-03f, 1.21372771e-02f, 6.95909781e-04f, 2.23214296e-03f, 3.42668802e-03f, 3.03867878e-03f,  // blk 20
    7.29370117e-03f, 7.31985923e-03f, 9.88769531e-03f, 4.67790896e-03f, 1.33231031e-02f, 7.57489877e-04f,
    2.23214296e-03f, 2.76402058e-03f, 2.38473085e-03f, 7.67735066e-03f, 4.04575886e-03f, 1.76391602e-02f,  // blk 21
    2.63977051e-03f, 1.67846680e-02f, 5.88008319e-04f, 2.23214296e-03f, 2.97764363e-03f, 2.45448528e-03f,  // blk 22
    1.17710661e-02f, 6.02940144e-03f, 1.70724057e-02f, 7.04084104e-03f, 1.73601415e-02f, 6.81195932e-04f,
    2.23214296e-03f, 3.03213927e-03f, 1.46484375e-03f, 4.43289615e-02f, 4.49044351e-03f, 2.40478516e-02f,  // blk 23
    9.88769531e-03f, 5.34602557e-04f                                                                        // sub_out, enc_proj
};

// V3 (multilingual) activation scales — calibrated on 16 utterances (en/de/it/fr),
// element-wise max across all calibration runs. V3 has ~100x larger subsampling
// activations than V2 due to different weight distributions.
static const float FP8_BAKED_ACT_SCALES_V3[218] = {
    1.98102687e-02f, 1.25976562e-01f, 3.76892090e-03f, 2.23214296e-03f, 4.52357717e-02f, 1.04544507e-02f,  // blk 0
    2.48883933e-01f, 3.33077572e-02f, 2.47209817e-01f, 2.71170475e-02f, 2.42571142e-02f, 2.08500447e-03f,
    2.23214296e-03f, 3.78199993e-03f, 8.70186929e-03f, 1.39421737e-02f, 7.32857827e-03f, 1.83279850e-02f,  // blk 1
    1.55639648e-02f, 2.44663786e-02f, 1.93459645e-03f, 2.23214296e-03f, 5.06155845e-03f, 7.56835938e-03f,  // blk 2
    2.64020655e-02f, 7.09751667e-03f, 2.61579249e-02f, 1.36893140e-02f, 2.55998876e-02f, 1.61852152e-03f,
    2.23214296e-03f, 4.89589153e-03f, 6.67463010e-03f, 8.24497789e-02f, 9.94873047e-03f, 2.43617464e-02f,  // blk 3
    1.67933870e-02f, 5.28738834e-02f, 1.48773193e-03f, 2.23214296e-03f, 4.85665444e-03f, 6.91005168e-03f,  // blk 4
    6.23256154e-02f, 9.04192217e-03f, 2.01416016e-02f, 1.53547013e-02f, 4.92117740e-02f, 1.58364431e-03f,
    2.23214296e-03f, 5.35801472e-03f, 6.66155154e-03f, 7.63811395e-02f, 7.29806069e-03f, 5.07114939e-02f,  // blk 5
    8.49696528e-03f, 5.05371094e-02f, 1.27846852e-03f, 2.23214296e-03f, 5.38853230e-03f, 6.06427854e-03f,  // blk 6
    5.92215396e-02f, 4.15257039e-03f, 2.52511166e-02f, 1.51192797e-02f, 6.21861033e-02f, 1.16729736e-03f,
    2.23214296e-03f, 3.88663146e-03f, 7.00160442e-03f, 5.96749447e-02f, 5.55419922e-03f, 3.05350162e-02f,  // blk 7
    2.44838167e-02f, 2.99421046e-02f, 1.52587891e-03f, 2.23214296e-03f, 6.24302449e-03f, 6.27790159e-03f,  // blk 8
    3.36739682e-02f, 8.84137861e-03f, 2.51813624e-02f, 9.69587080e-03f, 4.49218750e-02f, 1.02179393e-03f,
    2.23214296e-03f, 5.27518149e-03f, 5.26646199e-03f, 1.72293529e-01f, 4.32041707e-03f, 3.20870541e-02f,  // blk 9
    7.08879763e-03f, 5.49316406e-02f, 1.07574463e-03f, 2.23214296e-03f, 5.43648843e-03f, 4.44466714e-03f,  // blk 10
    4.96651791e-02f, 5.10515505e-03f, 3.13371941e-02f, 8.06971919e-03f, 1.20675221e-01f, 1.08664378e-03f,
    2.23214296e-03f, 8.28334223e-03f, 4.50352253e-03f, 4.24804688e-02f, 4.84357541e-03f, 3.11104916e-02f,  // blk 11
    1.09514510e-02f, 8.52399543e-02f, 1.12806051e-03f, 2.23214296e-03f, 8.58415850e-03f, 4.17872844e-03f,  // blk 12
    3.73883918e-02f, 3.01252096e-03f, 3.19998600e-02f, 8.15691240e-03f, 6.39299676e-02f, 1.30789622e-03f,
    2.23214296e-03f, 9.22502764e-03f, 5.20978635e-03f, 3.80510613e-02f, 4.75638267e-03f, 2.66462062e-02f,  // blk 13
    8.98960698e-03f, 4.39104363e-02f, 9.37325589e-04f, 2.23214296e-03f, 6.53948123e-03f, 4.93948814e-03f,  // blk 14
    2.93143131e-02f, 3.85611388e-03f, 2.70472933e-02f, 6.44792849e-03f, 4.20619436e-02f, 9.37325589e-04f,
    2.23214296e-03f, 5.13131265e-03f, 5.75038372e-03f, 3.70047428e-02f, 5.24030393e-03f, 2.80761719e-02f,  // blk 15
    6.69642864e-03f, 4.95256707e-02f, 1.66756765e-03f, 2.23214296e-03f, 8.18742998e-03f, 6.75746379e-03f,  // blk 16
    3.29066701e-02f, 7.32421875e-03f, 2.50941683e-02f, 1.20675219e-02f, 4.64215949e-02f, 1.35803223e-03f,
    2.23214296e-03f, 6.32585818e-03f, 6.36073528e-03f, 3.19126658e-02f, 5.38417278e-03f, 2.62974333e-02f,  // blk 17
    9.93129145e-03f, 2.97328401e-02f, 8.89914401e-04f, 2.23214296e-03f, 5.39289182e-03f, 6.20378787e-03f,  // blk 18
    4.44335938e-02f, 7.83429854e-03f, 3.05698942e-02f, 4.99180378e-03f, 5.29785156e-02f, 1.02887838e-03f,
    2.23214296e-03f, 7.71658774e-03f, 5.87245403e-03f, 3.54875848e-02f, 1.01318359e-02f, 2.87214015e-02f,  // blk 19
    1.58255436e-02f, 4.08761166e-02f, 1.10244751e-03f, 2.23214296e-03f, 5.41469036e-03f, 3.61633301e-03f,  // blk 20
    4.43638377e-02f, 1.41775953e-02f, 3.46679688e-02f, 1.40991211e-02f, 3.84347104e-02f, 1.36566162e-03f,
    2.23214296e-03f, 9.11167730e-03f, 3.08445515e-03f, 3.74581479e-02f, 1.10473633e-02f, 3.28194760e-02f,  // blk 21
    5.68934856e-03f, 3.90625000e-02f, 1.26102997e-03f, 2.23214296e-03f, 7.03648152e-03f, 3.17164836e-03f,  // blk 22
    7.44280145e-02f, 1.11955917e-02f, 7.89620504e-02f, 1.03846956e-02f, 3.16336490e-02f, 2.71388469e-03f,
    2.23214296e-03f, 8.97216797e-03f, 4.97000571e-03f, 3.55747752e-02f, 9.59995855e-03f, 4.79561947e-02f,  // blk 23
    1.76674104e+00f, 6.35964505e-04f                                                                        // sub_out, enc_proj
};

// =========================================================================
// CudaModel — encoder + decoder forward pass
// =========================================================================

// CUDA_CHECK is provided by common.h

// ---------------------------------------------------------------------------
// CudaModel::init
// ---------------------------------------------------------------------------

void CudaModel::init(const Weights& weights, cudaStream_t s, int max_mel_frames,
                     const char* fp8_path,
                     const void* fp8_prefetch, size_t fp8_prefetch_size) {
    w = &weights;
    stream = s;
    T_max = max_mel_frames / 8 + 10;  // encoder frames after 8x downsampling

    // CUTLASS workspace for FP16 (sub-conv, batched attention) and FP8 GEMMs
    cutlass_gemm_init(stream);
    cutlass_fp8_gemm_init(stream);

    // --- Pooled GPU allocation with phase-based buffer aliasing ---
    //
    // Subsampling buffers (sub_buf, mel_fp16) are only used at the start of
    // encode_gpu(). Conformer buffers (scores, ff_mid, etc.) are only used
    // in the subsequent conformer loop. Since their lifetimes don't overlap,
    // they share the same GPU memory region ("aliased region").
    //
    // Layout: [aliased region | persistent region]
    //   aliased  = max(subsampling_phase, conformer_phase)
    //   persistent = x, pos_enc, decoder/LSTM buffers, mel_fp32

    // Compute intermediate spatial dims for subsampling conv chain
    int H2 = (max_mel_frames + 2 * 1 - 3) / 2 + 1;
    int W2 = (128             + 2 * 1 - 3) / 2 + 1;  // 64
    int H3 = (H2              + 2 * 1 - 3) / 2 + 1;
    int W3 = (W2              + 2 * 1 - 3) / 2 + 1;  // 32

    size_t mel_fp32_elems = 128 * max_mel_frames;
    constexpr size_t ALIGN = 256;

    // Compute aligned total bytes for an array of half-element counts
    auto region_bytes = [](const size_t* sizes, int n) -> size_t {
        size_t off = 0;
        for (int i = 0; i < n; i++) {
            off = (off + 255) & ~(size_t)255;
            off += sizes[i] * sizeof(half);
        }
        return (off + 255) & ~(size_t)255;
    };

    // Subsampling-only buffers (dead after subsampling completes)
    size_t sub_sizes[] = {
        (size_t)(SUB_CHANNELS * H3 * W3),       // sub_buf[0]: peak at conv.2 output
        (size_t)(SUB_CHANNELS * H2 * W2),       // sub_buf[1]: peak at conv.0 output
        (size_t)(128 * max_mel_frames),         // mel_fp16
    };

    // Conformer-only buffers (unused during subsampling, no 'q' — uses q_u/q_v_buf)
    size_t conf_sizes[] = {
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

    // Persistent buffers (span both phases or used in decoder)
    size_t persist_sizes[] = {
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
        (size_t)w->config.d_output,             // joint_out
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[0]
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[1]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[0]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[1]
    };

    size_t sub_bytes  = region_bytes(sub_sizes, 3);
    size_t conf_bytes = region_bytes(conf_sizes, 16);
    size_t aliased_bytes = std::max(sub_bytes, conf_bytes);

    size_t persist_bytes = region_bytes(persist_sizes, 20);

    size_t pool_bytes = aliased_bytes + persist_bytes
                      + ALIGN + mel_fp32_elems * sizeof(float)
                      + ALIGN + 2 * sizeof(int);

    char* pool;
    CUDA_CHECK(cudaMalloc(&pool, pool_bytes));
    gpu_pool = pool;

    auto take = [](char*& cursor, size_t n) -> half* {
        cursor = (char*)(((uintptr_t)cursor + ALIGN - 1) & ~(ALIGN - 1));
        half* r = (half*)cursor;
        cursor += n * sizeof(half);
        return r;
    };

    // --- Phase-based buffer aliasing ---
    // Subsampling buffers and conformer scratch buffers share the same GPU
    // memory region. This is safe because subsampling runs first (converting
    // mel frames into encoder input x), and once x is written to the persistent
    // region the subsampling buffers are never touched again. The conformer
    // blocks then reuse that same memory for intermediates (ln_out, ff_mid,
    // qkv, scores, etc.).
    char* sub_cur = pool;
    sub_buf[0] = take(sub_cur, sub_sizes[0]);
    sub_buf[1] = take(sub_cur, sub_sizes[1]);
    mel_fp16   = take(sub_cur, sub_sizes[2]);

    char* conf_cur = pool;  // same base address — intentionally aliased
    ln_out     = take(conf_cur, conf_sizes[0]);
    ff_mid     = take(conf_cur, conf_sizes[1]);
    ff_out     = take(conf_cur, conf_sizes[2]);
    qkv        = take(conf_cur, conf_sizes[3]);
    k          = take(conf_cur, conf_sizes[4]);
    v          = take(conf_cur, conf_sizes[5]);
    pos_proj   = take(conf_cur, conf_sizes[6]);
    q_u        = take(conf_cur, conf_sizes[7]);
    q_v_buf    = take(conf_cur, conf_sizes[8]);
    scores     = take(conf_cur, conf_sizes[9]);
    pos_scores = take(conf_cur, conf_sizes[10]);
    attn_out   = take(conf_cur, conf_sizes[11]);
    mhsa_out   = take(conf_cur, conf_sizes[12]);
    conv_mid   = take(conf_cur, conf_sizes[13]);
    conv_glu   = take(conf_cur, conf_sizes[14]);
    conv_dw    = take(conf_cur, conf_sizes[15]);

    // --- Persistent region (after aliased region) ---
    char* pers_cur = pool + aliased_bytes;
    x              = take(pers_cur, persist_sizes[0]);
    pos_enc        = take(pers_cur, persist_sizes[1]);
    lstm_input     = take(pers_cur, persist_sizes[2]);
    lstm_gates     = take(pers_cur, persist_sizes[3]);
    lstm_h[0]      = take(pers_cur, persist_sizes[4]);
    lstm_h[1]      = take(pers_cur, persist_sizes[5]);
    lstm_c[0]      = take(pers_cur, persist_sizes[6]);
    lstm_c[1]      = take(pers_cur, persist_sizes[7]);
    lstm_h_out[0]  = take(pers_cur, persist_sizes[8]);
    lstm_h_out[1]  = take(pers_cur, persist_sizes[9]);
    lstm_c_out[0]  = take(pers_cur, persist_sizes[10]);
    lstm_c_out[1]  = take(pers_cur, persist_sizes[11]);
    enc_proj_all   = take(pers_cur, persist_sizes[12]);
    dec_proj_buf   = take(pers_cur, persist_sizes[13]);
    joint_act      = take(pers_cur, persist_sizes[14]);
    joint_out      = take(pers_cur, persist_sizes[15]);
    lstm_combined_w[0]    = take(pers_cur, persist_sizes[16]);
    lstm_combined_w[1]    = take(pers_cur, persist_sizes[17]);
    lstm_combined_bias[0] = take(pers_cur, persist_sizes[18]);
    lstm_combined_bias[1] = take(pers_cur, persist_sizes[19]);

    // mel_fp32 (float*) from persistent region, aligned
    pers_cur = (char*)(((uintptr_t)pers_cur + ALIGN - 1) & ~(ALIGN - 1));
    mel_fp32 = (float*)pers_cur;
    pers_cur += mel_fp32_elems * sizeof(float);

    // argmax_out (int[2]) from persistent region, aligned
    pers_cur = (char*)(((uintptr_t)pers_cur + ALIGN - 1) & ~(ALIGN - 1));
    argmax_out = (int*)pers_cur;
    pers_cur += 2 * sizeof(int);

    // Pre-compute position encoding for T_max (always needed, no FP16 dependency)
    generate_pos_encoding_gpu(pos_enc, T_max, D_MODEL, stream);

    // -----------------------------------------------------------------------
    // FP8 weight quantization — quantize all lt_gemm weight matrices to E4M3
    // -----------------------------------------------------------------------
    {
        // Compute total FP8 pool size (1 byte per element)
        size_t per_block = (size_t)D_MODEL * 3 * D_MODEL    // qkv_w
                         + (size_t)D_MODEL * D_FF            // ff1_w1
                         + (size_t)D_FF * D_MODEL            // ff1_w2
                         + (size_t)D_MODEL * D_FF            // ff2_w1
                         + (size_t)D_FF * D_MODEL            // ff2_w2
                         + (size_t)D_MODEL * D_MODEL          // pos_w
                         + (size_t)D_MODEL * D_MODEL          // out_w
                         + (size_t)D_CONV_PW * D_MODEL        // conv_pw1_w
                         + (size_t)D_MODEL * D_MODEL;         // conv_pw2_w
        size_t fp8_total = per_block * N_BLOCKS
                         + (size_t)SUB_CHANNELS * 16 * D_MODEL  // sub_out_w [4096,1024]
                         + (size_t)D_MODEL * D_JOINT              // enc_proj_w
                         + (size_t)4 * D_PRED * 2 * D_PRED * 2   // lstm_combined_w[0,1]
                         + (size_t)D_PRED * D_JOINT               // dec_proj_w
                         + (size_t)D_JOINT * w->config.d_output;  // out_proj_w
        // Add alignment padding (256 per weight) + scales + act buffer
        int n_fp8_ptrs = N_BLOCKS * 9 + 6;
        size_t fp8_pool_bytes = fp8_total + (size_t)n_fp8_ptrs * 256
                              + N_FP8_SCALES * sizeof(float) + 256
                              + (size_t)T_max * D_FF + 256            // act_buf
                              + sizeof(int) + 256                      // amax_buf
                              + N_FP8_ACT_SITES * sizeof(float) + 256  // act_site_scales
                              + N_FP8_ACT_SITES * sizeof(float) + 256; // alpha_products

        char* fp8p;
        CUDA_CHECK(cudaMalloc(&fp8p, fp8_pool_bytes));
        fp8_pool = fp8p;

        auto take8 = [&](size_t n) -> uint8_t* {
            fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
            uint8_t* r = (uint8_t*)fp8p;
            fp8p += n;
            return r;
        };

        // Per-block FP8 weights
        for (int b = 0; b < N_BLOCKS; b++) {
            fp8_qkv_w[b]     = take8(D_MODEL * 3 * D_MODEL);
            fp8_ff1_w1[b]    = take8(D_MODEL * D_FF);
            fp8_ff1_w2[b]    = take8(D_FF * D_MODEL);
            fp8_ff2_w1[b]    = take8(D_MODEL * D_FF);
            fp8_ff2_w2[b]    = take8(D_FF * D_MODEL);
            fp8_pos_w[b]     = take8(D_MODEL * D_MODEL);
            fp8_out_w[b]     = take8(D_MODEL * D_MODEL);
            fp8_conv_pw1_w[b] = take8(D_CONV_PW * D_MODEL);
            fp8_conv_pw2_w[b] = take8(D_MODEL * D_MODEL);
        }
        fp8_sub_out_w        = take8(SUB_CHANNELS * 16 * D_MODEL);
        fp8_enc_proj_w       = take8(D_MODEL * D_JOINT);
        fp8_lstm_combined_w[0] = take8(4 * D_PRED * 2 * D_PRED);
        fp8_lstm_combined_w[1] = take8(4 * D_PRED * 2 * D_PRED);
        fp8_dec_proj_w       = take8(D_PRED * D_JOINT);
        fp8_out_proj_w       = take8(D_JOINT * w->config.d_output);

        // Scales array
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_scales = (float*)fp8p;
        fp8p += N_FP8_SCALES * sizeof(float);

        // Activation scratch
        fp8_act_buf = take8(T_max * D_FF);

        // Per-site activation scale cache
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_act_site_scales = (float*)fp8p;
        fp8p += N_FP8_ACT_SITES * sizeof(float);

        // Pre-computed alpha products: alpha[i] = w_scale[w_idx] * act_scale[i]
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_alpha_products = (float*)fp8p;
        fp8p += N_FP8_ACT_SITES * sizeof(float);


        // ---------------------------------------------------------------------------
        // FP8 weight cache — paraketto-fp8.bin format:
        //   char[8]  magic   = "PRKTFP8\0"
        //   uint32   version = FP8_WEIGHTS_VERSION (2)
        //   uint32   model_version (0=v1, 3=v3)
        //   [fp8_pool blob: FP8 weights + scales, pool layout, single cudaMemcpy]
        //   [non-GEMM FP16 blob: LN, biases, conv_dw, embed, LSTM, decoder — packed]
        //
        // pool_weights_size is computed from the allocation above — no need to store it.
        // Try fp8_load first; if it fails, quantize from FP16 and save.
        // ---------------------------------------------------------------------------

        // Compute pool blob size (fp8_pool through end of fp8_scales, at their aligned offsets)
        size_t pool_weights_size = (size_t)((char*)(fp8_scales + N_FP8_SCALES) - (char*)fp8_pool);

        auto fp8_load = [&](const char* path) -> bool {
            const uint8_t* base = nullptr;
            size_t map_size = 0;
            void* mmap_ptr = nullptr;  // only set if we did our own mmap (need to munmap)

            if (fp8_prefetch) {
                // Use pre-populated mapping from background prefetch thread
                base = (const uint8_t*)fp8_prefetch;
                map_size = fp8_prefetch_size;
            } else {
                int fd = open(path, O_RDONLY);
                if (fd < 0) return false;
                struct stat st; fstat(fd, &st);
                map_size = (size_t)st.st_size;
                mmap_ptr = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd);
                if (mmap_ptr == MAP_FAILED) return false;
                madvise(mmap_ptr, map_size, MADV_SEQUENTIAL);
                base = (const uint8_t*)mmap_ptr;
            }

            // Validate header (16 bytes: magic + version + model_version)
            if (map_size < FP8_WEIGHTS_HEADER
                    || memcmp(base, "PRKTFP8", 7) != 0) {
                if (mmap_ptr) munmap(mmap_ptr, map_size);
                return false;
            }
            uint32_t version; memcpy(&version, base + 8, 4);
            uint32_t model_ver; memcpy(&model_ver, base + 12, 4);
            if (version != FP8_WEIGHTS_VERSION || (model_ver != 2 && model_ver != 3)) {
                if (mmap_ptr) munmap(mmap_ptr, map_size);
                return false;
            }
            if (map_size < FP8_WEIGHTS_HEADER + pool_weights_size) {
                if (mmap_ptr) munmap(mmap_ptr, map_size);
                return false;
            }

            // Single cudaMemcpy for fp8_pool (FP8 weights + scales at aligned offsets)
            CUDA_CHECK(cudaMemcpyAsync(fp8_pool, base + FP8_WEIGHTS_HEADER,
                                       pool_weights_size, cudaMemcpyHostToDevice, stream));

            // Non-GEMM FP16 weights — same order as generate_fp8_weights.py build_fp16_blob()
            const uint8_t* p = base + FP8_WEIGHTS_HEADER + pool_weights_size;
            size_t off = 0;
            auto ul16 = [&](half* dst, size_t n) {
                CUDA_CHECK(cudaMemcpyAsync(dst, p + off, n * sizeof(half),
                                           cudaMemcpyHostToDevice, stream));
                off += n * sizeof(half);
            };
            for (int i : {0, 2, 3, 5, 6}) {
                size_t wn = (i == 3 || i == 6) ? (size_t)SUB_CHANNELS * SUB_CHANNELS
                                                : SUB_CHANNELS * 9;
                ul16(weights.sub_conv[i].weight, wn);
                ul16(weights.sub_conv[i].bias,   SUB_CHANNELS);
            }
            ul16(weights.sub_out_b, D_MODEL);
            for (int b = 0; b < N_BLOCKS; b++) {
                auto& blk = weights.blocks[b];
                ul16(blk.ff1_ln_w,   D_MODEL); ul16(blk.ff1_ln_b,   D_MODEL);
                ul16(blk.mhsa_ln_w,  D_MODEL); ul16(blk.mhsa_ln_b,  D_MODEL);
                ul16(blk.pos_bias_u, (size_t)N_HEADS * HEAD_DIM);
                ul16(blk.pos_bias_v, (size_t)N_HEADS * HEAD_DIM);
                ul16(blk.conv_ln_w,  D_MODEL); ul16(blk.conv_ln_b,  D_MODEL);
                ul16(blk.conv_dw_w,  (size_t)D_MODEL * CONV_K);
                ul16(blk.conv_dw_b,  D_MODEL);
                ul16(blk.ff2_ln_w,   D_MODEL); ul16(blk.ff2_ln_b,   D_MODEL);
                ul16(blk.final_ln_w, D_MODEL); ul16(blk.final_ln_b, D_MODEL);
            }
            ul16(weights.embed_w,       (size_t)w->config.n_vocab * D_PRED);
            ul16(lstm_combined_w[0],    (size_t)4 * D_PRED * 2 * D_PRED);
            ul16(lstm_combined_w[1],    (size_t)4 * D_PRED * 2 * D_PRED);
            ul16(lstm_combined_bias[0], 4 * D_PRED);
            ul16(lstm_combined_bias[1], 4 * D_PRED);
            ul16(weights.dec_proj_w,    (size_t)D_PRED  * D_JOINT);
            ul16(weights.out_proj_w,    (size_t)D_JOINT * w->config.d_output);
            ul16(weights.enc_proj_b,    D_JOINT);
            ul16(weights.dec_proj_b,    D_JOINT);
            ul16(weights.out_proj_b,    w->config.d_output);

            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (mmap_ptr) munmap(mmap_ptr, map_size);

            size_t total_mb = (pool_weights_size + off) / (1024 * 1024);
            fprintf(stderr, "  loaded paraketto-fp8.bin from %s (%zu MB)\n", path, total_mb);
            return true;
        };

        // Load FP8 weights — fail hard if missing or corrupt.
        // Weight files are prepared offline (scripts/repack_weights.py) and hosted on HF.
        if (!fp8_path || !fp8_load(fp8_path)) {
            fprintf(stderr, "error: failed to load FP8 weights from '%s'\n"
                            "       Delete and re-download:  rm '%s' && make weights-fp8\n",
                    fp8_path ? fp8_path : "(null)", fp8_path ? fp8_path : "");
            std::exit(1);
        }

        // CUTLASS TN layout requires [n,k] row-major; transpose NN weights.
        // NT weights (conv_pw1, conv_pw2) are already [n,k] — skip them.
        {
            size_t max_sz = std::max({
                (size_t)D_MODEL * 3 * D_MODEL,       // qkv
                (size_t)D_MODEL * D_FF,               // ff w1
                (size_t)D_FF * D_MODEL,               // ff w2
                (size_t)SUB_CHANNELS * 16 * D_MODEL,  // sub_out
                (size_t)D_MODEL * D_JOINT              // enc_proj
            });
            void* transpose_tmp;
            CUDA_CHECK(cudaMalloc(&transpose_tmp, max_sz));
            for (int b = 0; b < N_BLOCKS; b++) {
                transpose_u8_inplace(fp8_qkv_w[b],  D_MODEL, 3 * D_MODEL, transpose_tmp, stream);
                transpose_u8_inplace(fp8_ff1_w1[b],  D_MODEL, D_FF,        transpose_tmp, stream);
                transpose_u8_inplace(fp8_ff1_w2[b],  D_FF,    D_MODEL,     transpose_tmp, stream);
                transpose_u8_inplace(fp8_ff2_w1[b],  D_MODEL, D_FF,        transpose_tmp, stream);
                transpose_u8_inplace(fp8_ff2_w2[b],  D_FF,    D_MODEL,     transpose_tmp, stream);
                transpose_u8_inplace(fp8_pos_w[b],   D_MODEL, D_MODEL,     transpose_tmp, stream);
                transpose_u8_inplace(fp8_out_w[b],   D_MODEL, D_MODEL,     transpose_tmp, stream);
                // conv_pw1, conv_pw2: NT weights, already [n,k] — skip
            }
            transpose_u8_inplace(fp8_sub_out_w,  SUB_CHANNELS * 16, D_MODEL, transpose_tmp, stream);
            transpose_u8_inplace(fp8_enc_proj_w, D_MODEL,           D_JOINT,  transpose_tmp, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaFree(transpose_tmp);
        }

        // Load baked activation scales (V2 or V3) and pre-compute alpha products
        static_assert(sizeof(FP8_BAKED_ACT_SCALES)    / sizeof(float) == N_FP8_ACT_SITES, "V2 scale count mismatch");
        static_assert(sizeof(FP8_BAKED_ACT_SCALES_V3) / sizeof(float) == N_FP8_ACT_SITES, "V3 scale count mismatch");
        const float* baked = (w->config.version == 3) ? FP8_BAKED_ACT_SCALES_V3
                                                      : FP8_BAKED_ACT_SCALES;
        CUDA_CHECK(cudaMemcpyAsync(fp8_act_site_scales, baked,
                                   N_FP8_ACT_SITES * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        fp8_compute_all_alphas(fp8_alpha_products, fp8_scales, fp8_act_site_scales,
                               N_BLOCKS, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// GEMM infrastructure removed — all GEMMs now use CUTLASS
// (FP8 via cutlass_gemm_fp8.h, FP16 via cutlass_gemm.h)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CudaModel::free
// ---------------------------------------------------------------------------

void CudaModel::free() {
    cutlass_gemm_free();
    cutlass_fp8_gemm_free();

    // All inference buffers are carved from a single pooled allocation
    if (gpu_pool) { cudaFree(gpu_pool); gpu_pool = nullptr; }
    if (fp8_pool) { cudaFree(fp8_pool); fp8_pool = nullptr; }
}

// ---------------------------------------------------------------------------
// FP8 GEMM wrapper — CUTLASS FP8 with per-site activation quantization
// ---------------------------------------------------------------------------

// Unified FP8 GEMM: quantize activation with pre-computed scales, run CUTLASS.
// All weights are [n,k] row-major (NN weights pre-transposed at init).
// Activation scales and alpha products are baked in at init time.
static void fp8_gemm_impl(cudaStream_t stream,
                           const half* X_fp16, int m, int k,
                           const uint8_t* W_fp8, int n,
                           half* Y,
                           uint8_t* act_buf, float* act_scale,
                           float* alpha_buf,
                           bool prequantized) {
    if (!prequantized)
        quantize_fp8_static(X_fp16, act_buf, act_scale, m * k, stream);
    cutlass_fp8_gemm(stream, act_buf, m, k, W_fp8, n, alpha_buf, Y);
}

// Position encoding generation moved to GPU kernel (generate_pos_encoding_gpu)

int CudaModel::encode_gpu(int T_mel) {
    // FP16 GEMM helper (sub-conv pointwise convolutions)
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        cutlass_gemm_nn(stream, X, m, k, W, n, Y);
    };

    // FP8 GEMM helpers — CUTLASS FP8 with baked activation scales
    auto gnn8 = [&](const half* X, int m, int k, const uint8_t* W, int n,
                     const float* /*ws*/, int si, half* Y, bool preq = false) {
        fp8_gemm_impl(stream, X, m, k, W, n, Y, fp8_act_buf,
                      &fp8_act_site_scales[si],
                      &fp8_alpha_products[si], preq);
    };
    auto gnn8_bias = [&](const half* X, int m, int k, const uint8_t* W, int n,
                          const float* /*ws*/, int si, const half* bias, half* Y,
                          bool preq = false) {
        fp8_gemm_impl(stream, X, m, k, W, n, Y, fp8_act_buf,
                      &fp8_act_site_scales[si],
                      &fp8_alpha_products[si], preq);
        bias_add_row_fp16(Y, bias, m, n, stream);
    };
    auto gnt8 = [&](const half* X, int m, int k, const uint8_t* W, int n,
                     const float* /*ws*/, int si, half* Y, bool preq = false) {
        fp8_gemm_impl(stream, X, m, k, W, n, Y, fp8_act_buf,
                      &fp8_act_site_scales[si],
                      &fp8_alpha_products[si], preq);
    };
    // Weight scale helper: fp8_scales[blk * 9 + offset]
    auto bscale = [&](int blk, int off) -> const float* { return &fp8_scales[blk * 9 + off]; };
    // Activation site index: blk * 9 + offset (0=ff1_w1, 1=ff1_w2, 2=qkv, 3=pos, 4=out,
    //                                          5=conv_pw1, 6=conv_pw2, 7=ff2_w1, 8=ff2_w2)
    auto asite = [](int blk, int off) -> int { return blk * 9 + off; };

    // 1. Cast mel to FP16, then transpose
    cast_fp32_to_fp16(mel_fp32, mel_fp16, 128 * T_mel, stream);
    transpose_fp16(mel_fp16, sub_buf[0], 128, T_mel, stream);

    // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
    //    Small sub-conv GEMMs stay FP16 (tiny weights, not worth FP8 overhead)
    int H = T_mel, W = 128;
    int H2 = (H + 2 * 1 - 3) / 2 + 1;
    int W2 = (W + 2 * 1 - 3) / 2 + 1;
    conv2d_fp16(sub_buf[0], w->sub_conv[0].weight, nullptr,
                sub_buf[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1, stream);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[0].bias, SUB_CHANNELS, H2 * W2, stream);
    H = H2; W = W2;

    conv2d_fp16(sub_buf[1], w->sub_conv[2].weight, w->sub_conv[2].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;
    gnn(w->sub_conv[3].weight, SUB_CHANNELS, SUB_CHANNELS, sub_buf[0], H * W, sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[3].bias, SUB_CHANNELS, H * W, stream);

    conv2d_fp16(sub_buf[1], w->sub_conv[5].weight, w->sub_conv[5].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;
    gnn(w->sub_conv[6].weight, SUB_CHANNELS, SUB_CHANNELS, sub_buf[0], H * W, sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[6].bias, SUB_CHANNELS, H * W, stream);

    int T = H;

    // reshape_chw_to_hcw + fused FP8 quantize: site 216 = N_BLOCKS*9 + 0 (sub_out activation)
    int sub_out_si = N_BLOCKS * 9 + 0;
    reshape_chw_to_hcw_fp8(sub_buf[1], sub_buf[0], fp8_act_buf,
                            &fp8_act_site_scales[sub_out_si],
                            SUB_CHANNELS, T, W, stream);
    // sub_out projection: FP8 (activation site: N_BLOCKS*9 + 0)
    const float* sub_out_scale = &fp8_scales[N_BLOCKS * 9 + 0];
    if (w->sub_out_b)
        gnn8_bias(sub_buf[0], T, SUB_CHANNELS * W, fp8_sub_out_w, D_MODEL, sub_out_scale, sub_out_si, w->sub_out_b, x, true);
    else
        gnn8(sub_buf[0], T, SUB_CHANNELS * W, fp8_sub_out_w, D_MODEL, sub_out_scale, sub_out_si, x, true);

    // 3. Position encoding
    half* pos_enc_T = pos_enc + (T_max - T) * D_MODEL;
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks — all lt_gemm-based GEMMs use FP8
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = w->blocks[blk];

        // --- FF1 (half-step residual) ---
        // layer_norm + fused FP8 quantize → asite(blk, 0) feeds ff1_w1
        layer_norm_fp8(x, b.ff1_ln_w, b.ff1_ln_b, ln_out, fp8_act_buf,
                       &fp8_act_site_scales[asite(blk, 0)], T, D_MODEL, NORM_EPS, stream);

        gnn8(ln_out, T, D_MODEL, fp8_ff1_w1[blk], D_FF, bscale(blk, 1), asite(blk, 0), ff_mid, true);
        // silu + fused FP8 quantize → asite(blk, 1) feeds ff1_w2
        silu_inplace_fp8(ff_mid, fp8_act_buf,
                         &fp8_act_site_scales[asite(blk, 1)], T * D_FF, stream);

        gnn8(ff_mid, T, D_FF, fp8_ff1_w2[blk], D_MODEL, bscale(blk, 2), asite(blk, 1), ff_out, true);
        // residual_add_layer_norm + fused FP8 quantize → asite(blk, 2) feeds qkv
        residual_add_layer_norm_fp8(x, ff_out, 0.5f,
            b.mhsa_ln_w, b.mhsa_ln_b, ln_out, fp8_act_buf,
            &fp8_act_site_scales[asite(blk, 2)], T, D_MODEL, NORM_EPS, stream);

        {
            // Fused QKV projection: FP8
            gnn8(ln_out, T, D_MODEL, fp8_qkv_w[blk], 3 * D_MODEL, bscale(blk, 0), asite(blk, 2), qkv, true);

            half* K_h = k;
            half* V_h = v;
            split_transpose_qkv_bias_fp16(qkv, b.pos_bias_u, b.pos_bias_v,
                                           q_u, q_v_buf, K_h, V_h,
                                           T, N_HEADS, HEAD_DIM, stream);

            // Position encoding projection: FP8 (NOT fused — precomputed data, per-block scales)
            half* pos_temp = pos_proj;
            gnn8(pos_enc_T, pos_len, D_MODEL, fp8_pos_w[blk], D_MODEL, bscale(blk, 5), asite(blk, 3), pos_temp, false);

            // Batched GEMMs stay FP16 (dynamic activations, not weight matrices)
            cutlass_batched_gemm_nt_ex(stream,
                q_v_buf, HEAD_DIM, (long long)T * HEAD_DIM,
                pos_temp, D_MODEL, (long long)HEAD_DIM,
                pos_scores, pos_len, (long long)T * pos_len,
                N_HEADS, T, pos_len, HEAD_DIM);

            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            cutlass_batched_gemm_nt(stream, q_u, K_h, scores,
                            N_HEADS, T, T, HEAD_DIM,
                            (long long)T * HEAD_DIM, (long long)T * HEAD_DIM, (long long)T * T);
            fused_score_softmax_fp16(scores, pos_scores, scores,
                                      N_HEADS, T, scale, stream);
            cutlass_batched_gemm_nn(stream, scores, V_h, attn_out,
                            N_HEADS, T, HEAD_DIM, T,
                            (long long)T * T, (long long)T * HEAD_DIM, (long long)T * HEAD_DIM);
            // transpose_0213 + fused FP8 quantize → asite(blk, 4) feeds out_w
            transpose_0213_fp8(attn_out, ff_out, fp8_act_buf,
                                &fp8_act_site_scales[asite(blk, 4)],
                                N_HEADS, T, HEAD_DIM, stream);

            // Output projection: FP8
            gnn8(ff_out, T, D_MODEL, fp8_out_w[blk], D_MODEL, bscale(blk, 6), asite(blk, 4), mhsa_out, true);
        }

        // residual_add_layer_norm + fused FP8 quantize → asite(blk, 5) feeds conv_pw1
        residual_add_layer_norm_fp8(x, mhsa_out, 1.0f,
            b.conv_ln_w, b.conv_ln_b, ln_out, fp8_act_buf,
            &fp8_act_site_scales[asite(blk, 5)], T, D_MODEL, NORM_EPS, stream);

        // Pointwise conv1 + GLU: FP8
        gnt8(ln_out, T, D_MODEL, fp8_conv_pw1_w[blk], D_CONV_PW, bscale(blk, 7), asite(blk, 5), conv_mid, true);
        glu_fp16(conv_mid, conv_glu, T, D_MODEL, stream);

        // depthwise_conv1d + SiLU + fused FP8 quantize → asite(blk, 6) feeds conv_pw2
        depthwise_conv1d_k9_silu_fp8(conv_glu, b.conv_dw_w, b.conv_dw_b,
                                      conv_dw, fp8_act_buf,
                                      &fp8_act_site_scales[asite(blk, 6)],
                                      T, D_MODEL, stream);

        // Pointwise conv2: FP8
        gnt8(conv_dw, T, D_MODEL, fp8_conv_pw2_w[blk], D_MODEL, bscale(blk, 8), asite(blk, 6), mhsa_out, true);

        // residual_add_layer_norm + fused FP8 quantize → asite(blk, 7) feeds ff2_w1
        residual_add_layer_norm_fp8(x, mhsa_out, 1.0f,
            b.ff2_ln_w, b.ff2_ln_b, ln_out, fp8_act_buf,
            &fp8_act_site_scales[asite(blk, 7)], T, D_MODEL, NORM_EPS, stream);

        // FF2: FP8
        gnn8(ln_out, T, D_MODEL, fp8_ff2_w1[blk], D_FF, bscale(blk, 3), asite(blk, 7), ff_mid, true);
        // silu + fused FP8 quantize → asite(blk, 8) feeds ff2_w2
        silu_inplace_fp8(ff_mid, fp8_act_buf,
                         &fp8_act_site_scales[asite(blk, 8)], T * D_FF, stream);

        gnn8(ff_mid, T, D_FF, fp8_ff2_w2[blk], D_MODEL, bscale(blk, 4), asite(blk, 8), ff_out, true);
        // Final LN: last block fuses FP8 for enc_proj (site 217 = N_BLOCKS*9 + 1)
        int enc_proj_si = N_BLOCKS * 9 + 1;
        if (blk == N_BLOCKS - 1)
            residual_add_layer_norm_fp8(x, ff_out, 0.5f,
                b.final_ln_w, b.final_ln_b, x, fp8_act_buf,
                &fp8_act_site_scales[enc_proj_si], T, D_MODEL, NORM_EPS, stream);
        else
            residual_add_layer_norm_fp16(x, ff_out, 0.5f,
                b.final_ln_w, b.final_ln_b, x, T, D_MODEL, NORM_EPS, stream);
    }

    // Encoder projection: FP8 (activation site: N_BLOCKS*9 + 1)
    {
        const float* enc_proj_scale = &fp8_scales[N_BLOCKS * 9 + 1];
        int epi = N_BLOCKS * 9 + 1;
        gnn8_bias(x, T, D_MODEL, fp8_enc_proj_w, D_JOINT, enc_proj_scale, epi, w->enc_proj_b, enc_proj_all, true);
    }

    return T;
}

// ---------------------------------------------------------------------------
// CudaModel::decoder_reset
// ---------------------------------------------------------------------------

void CudaModel::decoder_reset() {
    CUDA_CHECK(cudaMemsetAsync(lstm_h[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_h[1], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[1], 0, D_PRED * sizeof(half), stream));
}

// ---------------------------------------------------------------------------
// CudaModel::decode_step
// ---------------------------------------------------------------------------

half* CudaModel::decode_step(int enc_frame_idx, int prev_token) {
    // Decoder GEMMs use FP16 CUTLASS (N=1 single-vector GEMV)
    auto gnn_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        cutlass_gemm_nn_bias(stream, X, m, k, W, n, bias, Y);
    };
    auto gnt_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        cutlass_gemm_nt_bias(stream, X, m, k, W, n, bias, Y);
    };

    // 1. Embed + concat with h[0] for LSTM0 input: lstm_input = [embed; h[0]]
    embed_concat_fp16(w->embed_w, prev_token, lstm_h[0], lstm_input, D_PRED, stream);

    // 2. LSTM layer 0 — FP16 GEMM with combined [W_ih|W_hh] weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[0], 4 * D_PRED, lstm_combined_bias[0], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[0], lstm_h_out[0], lstm_c_out[0], D_PRED, stream);

    // 3. Concat h_out[0] with h[1] for LSTM1 input
    concat_vectors_fp16(lstm_h_out[0], lstm_h[1], lstm_input, D_PRED, stream);

    // 4. LSTM layer 1 — FP16 GEMM
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[1], 4 * D_PRED, lstm_combined_bias[1], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[1], lstm_h_out[1], lstm_c_out[1], D_PRED, stream);

    // 5. Joint network — FP16 GEMMs
    half* enc_proj_t = enc_proj_all + enc_frame_idx * D_JOINT;
    gnn_b(lstm_h_out[1], 1, D_PRED, w->dec_proj_w, D_JOINT, w->dec_proj_b, dec_proj_buf);
    add_relu_fp16(enc_proj_t, dec_proj_buf, joint_act, D_JOINT, stream);
    gnn_b(joint_act, 1, D_JOINT, w->out_proj_w, w->config.d_output, w->out_proj_b, joint_out);

    return joint_out;
}

// ---------------------------------------------------------------------------
// CudaModel::decoder_commit — swap LSTM state after non-blank token
// ---------------------------------------------------------------------------

void CudaModel::decoder_commit() {
    std::swap(lstm_h[0], lstm_h_out[0]);
    std::swap(lstm_c[0], lstm_c_out[0]);
    std::swap(lstm_h[1], lstm_h_out[1]);
    std::swap(lstm_c[1], lstm_c_out[1]);
}
