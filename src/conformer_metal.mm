// conformer_metal.mm — Metal inference for Parakeet conformer
//
// Mirrors conformer.cpp: init, encode_gpu, decode_step, decoder_commit.
// All data (activations + weights) lives in a single pooled MTLBuffer so
// every kernel dispatch uses one buffer with byte offsets.

#import <Metal/Metal.h>

#include "conformer_metal.h"
#include "metal_context.h"
#include "metal_context_impl.h"
#include "metal_kernels.h"
#include "metal_gemm.h"
#include "common_metal.h"
#include "mel_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using half_t = __fp16;

static constexpr uint32_t WEIGHTS_MAGIC = 0x544B5250;
static constexpr size_t   WEIGHTS_HEADER = 8;
static constexpr size_t   ALIGN = 256;
// NORM_EPS defined in common_metal.h

// ---------------------------------------------------------------------------
// Weight offset table — mirrors assign_weight_pointers in weights.cpp
// ---------------------------------------------------------------------------

struct WeightOffsets {
    struct SubConv { size_t weight, bias; };
    SubConv sub_conv[7];
    size_t sub_out_w, sub_out_b;

    struct Block {
        size_t ff1_ln_w, ff1_ln_b, ff1_w1, ff1_w2;
        size_t mhsa_ln_w, mhsa_ln_b;
        size_t q_w, k_w, v_w, pos_w;
        size_t pos_bias_u, pos_bias_v, out_w;
        size_t conv_ln_w, conv_ln_b;
        size_t conv_pw1_w, conv_dw_w, conv_dw_b, conv_pw2_w;
        size_t ff2_ln_w, ff2_ln_b, ff2_w1, ff2_w2;
        size_t final_ln_w, final_ln_b;
    } blocks[24];

    size_t embed_w;
    size_t lstm0_w_ih, lstm0_w_hh, lstm0_bias;
    size_t lstm1_w_ih, lstm1_w_hh, lstm1_bias;
    size_t enc_proj_w, enc_proj_b;
    size_t dec_proj_w, dec_proj_b;
    size_t out_proj_w, out_proj_b;
};

static size_t compute_weight_layout(WeightOffsets& wo, int n_vocab, int d_output) {
    size_t off = 0;
    auto take = [&](size_t& dst, size_t n) {
        off = (off + 255) & ~(size_t)255;
        dst = off;
        off += n * sizeof(half_t);
    };

    take(wo.sub_conv[0].weight, SUB_CHANNELS * 9);
    take(wo.sub_conv[0].bias,   SUB_CHANNELS);
    take(wo.sub_conv[2].weight, SUB_CHANNELS * 9);
    take(wo.sub_conv[2].bias,   SUB_CHANNELS);
    take(wo.sub_conv[3].weight, (size_t)SUB_CHANNELS * SUB_CHANNELS);
    take(wo.sub_conv[3].bias,   SUB_CHANNELS);
    take(wo.sub_conv[5].weight, SUB_CHANNELS * 9);
    take(wo.sub_conv[5].bias,   SUB_CHANNELS);
    take(wo.sub_conv[6].weight, (size_t)SUB_CHANNELS * SUB_CHANNELS);
    take(wo.sub_conv[6].bias,   SUB_CHANNELS);
    take(wo.sub_out_w,          (size_t)SUB_CHANNELS * 16 * D_MODEL);
    take(wo.sub_out_b,          D_MODEL);

    for (int i = 0; i < N_BLOCKS; i++) {
        auto& b = wo.blocks[i];
        take(b.ff1_ln_w, D_MODEL); take(b.ff1_ln_b, D_MODEL);
        take(b.ff1_w1, (size_t)D_MODEL * D_FF);
        take(b.ff1_w2, (size_t)D_FF * D_MODEL);
        take(b.mhsa_ln_w, D_MODEL); take(b.mhsa_ln_b, D_MODEL);
        take(b.q_w, (size_t)D_MODEL * D_MODEL);
        take(b.k_w, (size_t)D_MODEL * D_MODEL);
        take(b.v_w, (size_t)D_MODEL * D_MODEL);
        take(b.pos_w, (size_t)D_MODEL * D_MODEL);
        take(b.pos_bias_u, (size_t)N_HEADS * HEAD_DIM);
        take(b.pos_bias_v, (size_t)N_HEADS * HEAD_DIM);
        take(b.out_w, (size_t)D_MODEL * D_MODEL);
        take(b.conv_ln_w, D_MODEL); take(b.conv_ln_b, D_MODEL);
        take(b.conv_pw1_w, (size_t)D_CONV_PW * D_MODEL);
        take(b.conv_dw_w, (size_t)D_MODEL * CONV_K);
        take(b.conv_dw_b, D_MODEL);
        take(b.conv_pw2_w, (size_t)D_MODEL * D_MODEL);
        take(b.ff2_ln_w, D_MODEL); take(b.ff2_ln_b, D_MODEL);
        take(b.ff2_w1, (size_t)D_MODEL * D_FF);
        take(b.ff2_w2, (size_t)D_FF * D_MODEL);
        take(b.final_ln_w, D_MODEL); take(b.final_ln_b, D_MODEL);
    }

    take(wo.embed_w, (size_t)n_vocab * D_PRED);
    take(wo.lstm0_w_ih, (size_t)4*D_PRED*D_PRED); take(wo.lstm0_w_hh, (size_t)4*D_PRED*D_PRED);
    take(wo.lstm0_bias, 8*D_PRED);
    take(wo.lstm1_w_ih, (size_t)4*D_PRED*D_PRED); take(wo.lstm1_w_hh, (size_t)4*D_PRED*D_PRED);
    take(wo.lstm1_bias, 8*D_PRED);
    take(wo.enc_proj_w, (size_t)D_MODEL*D_JOINT); take(wo.enc_proj_b, D_JOINT);
    take(wo.dec_proj_w, (size_t)D_PRED*D_JOINT);  take(wo.dec_proj_b, D_JOINT);
    take(wo.out_proj_w, (size_t)D_JOINT*d_output); take(wo.out_proj_b, d_output);

    return (off + 255) & ~(size_t)255;
}

// Weight offsets are relative to the file data. After loading, we add
// weight_base_off to get the offset within the unified gpu_pool.
static WeightOffsets g_wo;
static size_t g_weight_base = 0;  // byte offset of weights region in gpu_pool

// Convenience: weight offset in the unified pool
#define WF(field) (g_weight_base + g_wo.field)

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

void MetalModel::init(const char* weights_path, int max_mel_frames) {
    T_max = max_mel_frames / 8 + 10;

    // Load shaders
    auto read_file = [](const char* path) -> std::string {
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open shader: %s\n", path); exit(1); }
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        std::string s(len, '\0'); fread(&s[0], 1, len, f); fclose(f);
        return s;
    };
    std::string src = read_file("metal/kernels.metal");
    src += "\n";
    src += read_file("metal/gemm.metal");
    ctx.load_shaders(src.c_str(), "paraketto");
    metal_gemm_init(ctx);

    // Read weight file header
    int fd = open(weights_path, O_RDONLY);
    METAL_CHECK(fd >= 0, "Cannot open weights");
    struct stat st; fstat(fd, &st);
    size_t file_size = st.st_size;
    void* mmap_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    METAL_CHECK(mmap_ptr != MAP_FAILED, "mmap failed"); close(fd);

    const uint32_t* hdr = (const uint32_t*)mmap_ptr;
    METAL_CHECK(hdr[0] == WEIGHTS_MAGIC, "Bad magic");
    if (hdr[1] == 3) { n_vocab = 8193; d_output = 8198; blank_id = 8192; }

    weight_bytes = compute_weight_layout(g_wo, n_vocab, d_output);

    // Compute activation buffer layout
    int H2 = (max_mel_frames + 2 - 3) / 2 + 1;
    int W2 = (128 + 2 - 3) / 2 + 1;
    int H3 = (H2 + 2 - 3) / 2 + 1;
    int W3 = (W2 + 2 - 3) / 2 + 1;
    size_t mel_fp32_elems = 128 * max_mel_frames;
    size_t frames_fp32_elems = (size_t)max_mel_frames * N_FFT;  // [n_frames, 512]
    size_t mel_raw_elems = (size_t)max_mel_frames * N_MELS;     // [n_frames, 128]

    auto rbytes = [](const size_t* s, int n) -> size_t {
        size_t o = 0;
        for (int i = 0; i < n; i++) { o = (o+255)&~(size_t)255; o += s[i]*sizeof(half_t); }
        return (o+255)&~(size_t)255;
    };

    size_t sub_s[] = { (size_t)(SUB_CHANNELS*H3*W3), (size_t)(SUB_CHANNELS*H2*W2), (size_t)(128*max_mel_frames) };
    size_t conf_s[] = {
        (size_t)(T_max*D_MODEL), (size_t)(T_max*D_FF), (size_t)(T_max*D_MODEL),
        (size_t)(T_max*3*D_MODEL),
        (size_t)(N_HEADS*T_max*HEAD_DIM), (size_t)(N_HEADS*T_max*HEAD_DIM),
        (size_t)(N_HEADS*T_max*HEAD_DIM), (size_t)((2*T_max)*D_MODEL),
        (size_t)(N_HEADS*T_max*HEAD_DIM), (size_t)(N_HEADS*T_max*HEAD_DIM),
        (size_t)(N_HEADS*T_max*T_max), (size_t)(N_HEADS*T_max*(2*T_max)),
        (size_t)(N_HEADS*T_max*HEAD_DIM), (size_t)(T_max*D_MODEL),
        (size_t)(T_max*D_CONV_PW), (size_t)(T_max*D_MODEL), (size_t)(T_max*D_MODEL)
    };
    size_t pers_s[] = {
        (size_t)(T_max*D_MODEL), (size_t)((2*T_max)*D_MODEL),
        (size_t)(2*D_PRED), (size_t)(4*D_PRED),
        (size_t)D_PRED,(size_t)D_PRED,(size_t)D_PRED,(size_t)D_PRED,
        (size_t)D_PRED,(size_t)D_PRED,(size_t)D_PRED,(size_t)D_PRED,
        (size_t)(T_max*D_JOINT),(size_t)D_JOINT,(size_t)D_JOINT,(size_t)d_output,
        (size_t)(4*D_PRED*2*D_PRED),(size_t)(4*D_PRED*2*D_PRED),
        (size_t)(4*D_PRED),(size_t)(4*D_PRED)
    };

    size_t aliased = std::max(rbytes(sub_s,3), rbytes(conf_s,17));
    size_t persist = rbytes(pers_s, 20);
    size_t qkv_elem = (size_t)D_MODEL * 3 * D_MODEL;
    size_t qkv_blk = (qkv_elem * sizeof(half_t) + ALIGN - 1) & ~(ALIGN - 1);

    size_t act_bytes = aliased + persist + (size_t)N_BLOCKS * qkv_blk
                     + ALIGN + mel_fp32_elems * sizeof(float)
                     + ALIGN + frames_fp32_elems * sizeof(float)
                     + ALIGN + mel_raw_elems * sizeof(float)
                     + ALIGN + 2 * sizeof(int);

    // Unified pool: [activations | weights]
    g_weight_base = (act_bytes + ALIGN - 1) & ~(ALIGN - 1);
    gpu_pool_bytes = g_weight_base + weight_bytes;
    gpu_pool_handle = ctx.alloc_shared(gpu_pool_bytes);

    // Copy weights into pool
    char* pool = (char*)ctx.buffer_contents(gpu_pool_handle);
    memcpy(pool + g_weight_base, (const char*)mmap_ptr + WEIGHTS_HEADER, weight_bytes);
    munmap(mmap_ptr, file_size);
    fprintf(stderr, "[metal] weights: %.1f MB, pool: %.1f MB\n",
            weight_bytes/(1024.0*1024.0), gpu_pool_bytes/(1024.0*1024.0));

    // Assign activation offsets
    auto take = [](size_t& c, size_t n) -> size_t {
        c = (c+255)&~(size_t)255; size_t r = c; c += n*sizeof(half_t); return r;
    };

    size_t sc = 0;
    sub_buf_off[0] = take(sc, sub_s[0]);
    sub_buf_off[1] = take(sc, sub_s[1]);
    mel_fp16_off   = take(sc, sub_s[2]);

    size_t cc = 0;
    ln_out_off   = take(cc, conf_s[0]);  ff_mid_off   = take(cc, conf_s[1]);
    ff_out_off   = take(cc, conf_s[2]);  qkv_off      = take(cc, conf_s[3]);
    q_off        = take(cc, conf_s[4]);  k_off        = take(cc, conf_s[5]);
    v_off        = take(cc, conf_s[6]);  pos_proj_off = take(cc, conf_s[7]);
    q_u_off      = take(cc, conf_s[8]);  q_v_buf_off  = take(cc, conf_s[9]);
    scores_off   = take(cc, conf_s[10]); pos_scores_off = take(cc, conf_s[11]);
    attn_out_off = take(cc, conf_s[12]); mhsa_out_off = take(cc, conf_s[13]);
    conv_mid_off = take(cc, conf_s[14]); conv_glu_off = take(cc, conf_s[15]);
    conv_dw_off  = take(cc, conf_s[16]);

    size_t pc = aliased;
    x_off = take(pc, pers_s[0]); pos_enc_off = take(pc, pers_s[1]);
    lstm_input_off = take(pc, pers_s[2]); lstm_gates_off = take(pc, pers_s[3]);
    lstm_h_off[0] = take(pc, pers_s[4]); lstm_h_off[1] = take(pc, pers_s[5]);
    lstm_c_off[0] = take(pc, pers_s[6]); lstm_c_off[1] = take(pc, pers_s[7]);
    lstm_h_out_off[0] = take(pc, pers_s[8]); lstm_h_out_off[1] = take(pc, pers_s[9]);
    lstm_c_out_off[0] = take(pc, pers_s[10]); lstm_c_out_off[1] = take(pc, pers_s[11]);
    enc_proj_all_off = take(pc, pers_s[12]);
    dec_proj_buf_off = take(pc, pers_s[13]); joint_act_off = take(pc, pers_s[14]);
    joint_out_off = take(pc, pers_s[15]);
    lstm_combined_w_off[0] = take(pc, pers_s[16]); lstm_combined_w_off[1] = take(pc, pers_s[17]);
    lstm_combined_bias_off[0] = take(pc, pers_s[18]); lstm_combined_bias_off[1] = take(pc, pers_s[19]);

    for (int b = 0; b < N_BLOCKS; b++) {
        pc = (pc+ALIGN-1)&~(ALIGN-1); qkv_w_off[b] = pc; pc += qkv_elem*sizeof(half_t);
    }
    pc = (pc+ALIGN-1)&~(ALIGN-1); mel_fp32_off = pc; pc += mel_fp32_elems*sizeof(float);
    pc = (pc+ALIGN-1)&~(ALIGN-1); frames_fp32_off = pc; pc += frames_fp32_elems*sizeof(float);
    pc = (pc+ALIGN-1)&~(ALIGN-1); mel_raw_off = pc; pc += mel_raw_elems*sizeof(float);
    pc = (pc+ALIGN-1)&~(ALIGN-1); argmax_out_off = pc;

    // Pre-concatenate QKV weights (CPU, unified memory)
    for (int b = 0; b < N_BLOCKS; b++) {
        half_t* dst = (half_t*)(pool + qkv_w_off[b]);
        const half_t* qs = (const half_t*)(pool + WF(blocks[b].q_w));
        const half_t* ks = (const half_t*)(pool + WF(blocks[b].k_w));
        const half_t* vs = (const half_t*)(pool + WF(blocks[b].v_w));
        for (int r = 0; r < D_MODEL; r++) {
            memcpy(dst + r*3*D_MODEL,            qs + r*D_MODEL, D_MODEL*sizeof(half_t));
            memcpy(dst + r*3*D_MODEL + D_MODEL,  ks + r*D_MODEL, D_MODEL*sizeof(half_t));
            memcpy(dst + r*3*D_MODEL + 2*D_MODEL, vs + r*D_MODEL, D_MODEL*sizeof(half_t));
        }
    }

    // Pre-combine LSTM weights
    {
        size_t wih[2] = { WF(lstm0_w_ih), WF(lstm1_w_ih) };
        size_t whh[2] = { WF(lstm0_w_hh), WF(lstm1_w_hh) };
        size_t bis[2] = { WF(lstm0_bias), WF(lstm1_bias) };
        for (int l = 0; l < 2; l++) {
            half_t* dw = (half_t*)(pool + lstm_combined_w_off[l]);
            const half_t* ih = (const half_t*)(pool + wih[l]);
            const half_t* hh = (const half_t*)(pool + whh[l]);
            for (int r = 0; r < 4*D_PRED; r++) {
                memcpy(dw + r*2*D_PRED,        ih + r*D_PRED, D_PRED*sizeof(half_t));
                memcpy(dw + r*2*D_PRED+D_PRED, hh + r*D_PRED, D_PRED*sizeof(half_t));
            }
            half_t* db = (half_t*)(pool + lstm_combined_bias_off[l]);
            const half_t* sb = (const half_t*)(pool + bis[l]);
            for (int i = 0; i < 4*D_PRED; i++)
                db[i] = (half_t)((float)sb[i] + (float)sb[4*D_PRED + i]);
        }
    }

    // Position encoding
    {
        id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        metal_generate_pos_encoding_gpu(ctx, (__bridge void*)e, gpu_pool_handle,
                                        pos_enc_off, T_max, D_MODEL);
        [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    }

    // Mel filterbank (GPU)
    metal_mel_init_filterbank(ctx, MEL_FILTERBANK, N_MEL_ENTRIES);
}

void MetalModel::free() {
    if (gpu_pool_handle) { ctx.free_buffer(gpu_pool_handle); gpu_pool_handle = nullptr; }
    metal_gemm_free();
}

// ---------------------------------------------------------------------------
// GPU mel spectrogram: windowed frames → FFT → mel filterbank → normalize
// ---------------------------------------------------------------------------

void MetalModel::compute_mel_gpu(int n_frames, int n_valid) {
    void* P = gpu_pool_handle;
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    void* E = (__bridge void*)enc;

    // FFT + mel filterbank + log: frames[n_frames, 512] → mel_raw[n_frames, 128]
    metal_fft512_mel_log(ctx, E, P, frames_fp32_off, mel_raw_off, n_frames);

    // Normalize + transpose: mel_raw[n_frames, 128] → mel_fp32[128, n_valid]
    metal_mel_normalize(ctx, E, P, mel_raw_off, mel_fp32_off, n_frames, n_valid);

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
}

// ---------------------------------------------------------------------------
// Encode: mel → encoder output [T, D_MODEL]
// ---------------------------------------------------------------------------

int MetalModel::encode_gpu(int T_mel) {
    void* P = gpu_pool_handle;
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    void* E = (__bridge void*)enc;

    // 1. Cast + transpose mel
    metal_cast_fp32_to_fp16(ctx, E, P, mel_fp32_off, mel_fp16_off, 128 * T_mel);
    metal_transpose_fp16(ctx, E, P, mel_fp16_off, sub_buf_off[0], 128, T_mel);

    // 2. Subsampling convolutions
    int H = T_mel, W = 128;
    int H2 = (H+2-3)/2+1, W2 = (W+2-3)/2+1;

    metal_conv2d_fp16(ctx, E, P, sub_buf_off[0], WF(sub_conv[0].weight), SIZE_MAX,
                      sub_buf_off[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1);
    metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[0].bias), SUB_CHANNELS, H2*W2);
    H = H2; W = W2;

    metal_conv2d_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[2].weight), WF(sub_conv[2].bias),
                      sub_buf_off[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS);
    H = (H+2-3)/2+1; W = (W+2-3)/2+1;
    metal_gemm_nn(ctx, E, P, WF(sub_conv[3].weight), SUB_CHANNELS, SUB_CHANNELS,
                  sub_buf_off[0], H*W, sub_buf_off[1]);
    metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[3].bias), SUB_CHANNELS, H*W);

    metal_conv2d_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[5].weight), WF(sub_conv[5].bias),
                      sub_buf_off[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS);
    H = (H+2-3)/2+1; W = (W+2-3)/2+1;
    metal_gemm_nn(ctx, E, P, WF(sub_conv[6].weight), SUB_CHANNELS, SUB_CHANNELS,
                  sub_buf_off[0], H*W, sub_buf_off[1]);
    metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[6].bias), SUB_CHANNELS, H*W);

    int T = H;
    metal_reshape_chw_to_hcw_fp16(ctx, E, P, sub_buf_off[1], sub_buf_off[0], SUB_CHANNELS, T, W);
    metal_gemm_nn_bias(ctx, E, P, sub_buf_off[0], T, SUB_CHANNELS*W,
                       WF(sub_out_w), D_MODEL, WF(sub_out_b), x_off);

    // 3. Position encoding slice
    size_t pos_enc_T = pos_enc_off + (size_t)(T_max - T) * D_MODEL * sizeof(half_t);
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = g_wo.blocks[blk];

        // FF1
        metal_layer_norm_fp16(ctx, E, P, x_off, WF(blocks[blk].ff1_ln_w),
                              WF(blocks[blk].ff1_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);
        metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].ff1_w1), D_FF, ff_mid_off);
        metal_silu_inplace_fp16(ctx, E, P, ff_mid_off, T * D_FF);
        metal_gemm_nn(ctx, E, P, ff_mid_off, T, D_FF, WF(blocks[blk].ff1_w2), D_MODEL, ff_out_off);
        metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, ff_out_off, 0.5f,
            WF(blocks[blk].mhsa_ln_w), WF(blocks[blk].mhsa_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);

        // MHSA: QKV projection
        metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, qkv_w_off[blk], 3*D_MODEL, qkv_off);
        metal_split_transpose_qkv_bias_fp16(ctx, E, P, qkv_off,
            WF(blocks[blk].pos_bias_u), WF(blocks[blk].pos_bias_v),
            q_u_off, q_v_buf_off, k_off, v_off, T, N_HEADS, HEAD_DIM);

        // Position encoding projection
        metal_gemm_nn(ctx, E, P, pos_enc_T, pos_len, D_MODEL,
                      WF(blocks[blk].pos_w), D_MODEL, pos_proj_off);

        // Position scores: batched NT with explicit ld
        metal_batched_gemm_nt_ex(ctx, E, P,
            q_v_buf_off, HEAD_DIM, (int64_t)T * HEAD_DIM,
            pos_proj_off, D_MODEL, (int64_t)HEAD_DIM,
            pos_scores_off, pos_len, (int64_t)T * pos_len,
            N_HEADS, T, pos_len, HEAD_DIM);

        float scale = 1.0f / sqrtf((float)HEAD_DIM);

        // Content scores: batched NT q_u @ K^T
        metal_batched_gemm_nt(ctx, E, P, q_u_off, k_off, scores_off,
            N_HEADS, T, T, HEAD_DIM,
            (int64_t)T*HEAD_DIM, (int64_t)T*HEAD_DIM, (int64_t)T*T);

        // Fused softmax
        metal_fused_score_softmax_fp16(ctx, E, P, scores_off, pos_scores_off,
                                        scores_off, N_HEADS, T, scale);

        // Weighted sum: scores @ V
        metal_batched_gemm_nn(ctx, E, P, scores_off, v_off, attn_out_off,
            N_HEADS, T, HEAD_DIM, T,
            (int64_t)T*T, (int64_t)T*HEAD_DIM, (int64_t)T*HEAD_DIM);

        // Transpose + output projection
        metal_transpose_0213_fp16(ctx, E, P, attn_out_off, ff_out_off, N_HEADS, T, HEAD_DIM);
        metal_gemm_nn(ctx, E, P, ff_out_off, T, D_MODEL, WF(blocks[blk].out_w), D_MODEL, mhsa_out_off);

        // Residual + LN for conv
        metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, mhsa_out_off, 1.0f,
            WF(blocks[blk].conv_ln_w), WF(blocks[blk].conv_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);

        // Conv module
        metal_gemm_nt(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].conv_pw1_w), D_CONV_PW, conv_mid_off);
        metal_glu_fp16(ctx, E, P, conv_mid_off, conv_glu_off, T, D_MODEL);
        metal_depthwise_conv1d_k9_silu_fp16(ctx, E, P, conv_glu_off,
            WF(blocks[blk].conv_dw_w), WF(blocks[blk].conv_dw_b), conv_dw_off, T, D_MODEL);
        metal_gemm_nt(ctx, E, P, conv_dw_off, T, D_MODEL, WF(blocks[blk].conv_pw2_w), D_MODEL, mhsa_out_off);

        // Residual + LN for FF2
        metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, mhsa_out_off, 1.0f,
            WF(blocks[blk].ff2_ln_w), WF(blocks[blk].ff2_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);

        // FF2
        metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].ff2_w1), D_FF, ff_mid_off);
        metal_silu_inplace_fp16(ctx, E, P, ff_mid_off, T * D_FF);
        metal_gemm_nn(ctx, E, P, ff_mid_off, T, D_FF, WF(blocks[blk].ff2_w2), D_MODEL, ff_out_off);
        metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, ff_out_off, 0.5f,
            WF(blocks[blk].final_ln_w), WF(blocks[blk].final_ln_b), x_off, T, D_MODEL, NORM_EPS);
    }

    // Encoder projection
    metal_gemm_nn_bias(ctx, E, P, x_off, T, D_MODEL,
                       WF(enc_proj_w), D_JOINT, WF(enc_proj_b), enc_proj_all_off);

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return T;
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

void MetalModel::decoder_reset() {
    char* pool = (char*)ctx.buffer_contents(gpu_pool_handle);
    for (int i = 0; i < 2; i++) {
        memset(pool + lstm_h_off[i], 0, D_PRED * sizeof(half_t));
        memset(pool + lstm_c_off[i], 0, D_PRED * sizeof(half_t));
    }
}

int MetalModel::decode_step(int enc_frame_idx, int prev_token) {
    void* P = gpu_pool_handle;
    id<MTLCommandBuffer> cmd = [ctx.impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    void* E = (__bridge void*)enc;

    // 1. Embed + concat
    metal_embed_concat_fp16(ctx, E, P, WF(embed_w), prev_token,
                            lstm_h_off[0], lstm_input_off, D_PRED);

    // 2. LSTM layer 0
    metal_gemm_nt_bias(ctx, E, P, lstm_input_off, 1, 2*D_PRED,
                       lstm_combined_w_off[0], 4*D_PRED,
                       lstm_combined_bias_off[0], lstm_gates_off);
    metal_lstm_cell_fp16(ctx, E, P, lstm_gates_off, lstm_c_off[0],
                         lstm_h_out_off[0], lstm_c_out_off[0], D_PRED);

    // 3. Concat + LSTM layer 1
    metal_concat_vectors_fp16(ctx, E, P, lstm_h_out_off[0], lstm_h_off[1],
                              lstm_input_off, D_PRED);
    metal_gemm_nt_bias(ctx, E, P, lstm_input_off, 1, 2*D_PRED,
                       lstm_combined_w_off[1], 4*D_PRED,
                       lstm_combined_bias_off[1], lstm_gates_off);
    metal_lstm_cell_fp16(ctx, E, P, lstm_gates_off, lstm_c_off[1],
                         lstm_h_out_off[1], lstm_c_out_off[1], D_PRED);

    // 4. Joint network
    size_t enc_proj_t = enc_proj_all_off + (size_t)enc_frame_idx * D_JOINT * sizeof(half_t);
    metal_gemm_nn_bias(ctx, E, P, lstm_h_out_off[1], 1, D_PRED,
                       WF(dec_proj_w), D_JOINT, WF(dec_proj_b), dec_proj_buf_off);
    metal_add_relu_fp16(ctx, E, P, enc_proj_t, dec_proj_buf_off, joint_act_off, D_JOINT);
    metal_gemm_nn_bias(ctx, E, P, joint_act_off, 1, D_JOINT,
                       WF(out_proj_w), d_output, WF(out_proj_b), joint_out_off);

    // 5. Argmax
    metal_dual_argmax_fp16(ctx, E, P, joint_out_off, argmax_out_off, n_vocab, d_output);

    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

    // Read back argmax results (unified memory — direct access)
    char* pool = (char*)ctx.buffer_contents(gpu_pool_handle);
    int* argmax = (int*)(pool + argmax_out_off);
    return argmax[0];  // token (caller reads argmax[1] for duration separately)
}

void MetalModel::decoder_commit() {
    char* pool = (char*)ctx.buffer_contents(gpu_pool_handle);
    for (int i = 0; i < 2; i++) {
        memcpy(pool + lstm_h_off[i], pool + lstm_h_out_off[i], D_PRED * sizeof(half_t));
        memcpy(pool + lstm_c_off[i], pool + lstm_c_out_off[i], D_PRED * sizeof(half_t));
    }
}

// ---------------------------------------------------------------------------
// Profiled encode: separate command buffer per phase for GPU timing
// ---------------------------------------------------------------------------

// Helper: encode a phase in its own command buffer, return GPU time in ms
struct PhaseTimer {
    MetalContext& ctx;
    double gpu_ms = 0;

    explicit PhaseTimer(MetalContext& c) : ctx(c) {}

    id<MTLComputeCommandEncoder> begin() {
        cmd = [ctx.impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        return enc;
    }
    void end(id<MTLComputeCommandEncoder> enc) {
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0;
    }
private:
    id<MTLCommandBuffer> cmd = nil;
};

int MetalModel::encode_gpu_profile(int T_mel) {
    void* P = gpu_pool_handle;

    fprintf(stderr, "\n=== Metal Encoder Profile (T_mel=%d) ===\n", T_mel);
    double total_ms = 0;

    // Phase: Mel cast + transpose
    {
        PhaseTimer pt{ctx};
        auto enc = pt.begin(); void* E = (__bridge void*)enc;
        metal_cast_fp32_to_fp16(ctx, E, P, mel_fp32_off, mel_fp16_off, 128 * T_mel);
        metal_transpose_fp16(ctx, E, P, mel_fp16_off, sub_buf_off[0], 128, T_mel);
        pt.end(enc);
        fprintf(stderr, "  mel_prep:     %6.2f ms\n", pt.gpu_ms);
        total_ms += pt.gpu_ms;
    }

    // Phase: Subsampling convolutions
    int H = T_mel, W = 128;
    int H2, W2, T;
    {
        PhaseTimer pt{ctx};
        auto enc = pt.begin(); void* E = (__bridge void*)enc;

        H2 = (H+2-3)/2+1; W2 = (W+2-3)/2+1;
        metal_conv2d_fp16(ctx, E, P, sub_buf_off[0], WF(sub_conv[0].weight), SIZE_MAX,
                          sub_buf_off[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1);
        metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[0].bias), SUB_CHANNELS, H2*W2);
        H = H2; W = W2;

        metal_conv2d_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[2].weight), WF(sub_conv[2].bias),
                          sub_buf_off[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS);
        H = (H+2-3)/2+1; W = (W+2-3)/2+1;
        metal_gemm_nn(ctx, E, P, WF(sub_conv[3].weight), SUB_CHANNELS, SUB_CHANNELS,
                      sub_buf_off[0], H*W, sub_buf_off[1]);
        metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[3].bias), SUB_CHANNELS, H*W);

        metal_conv2d_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[5].weight), WF(sub_conv[5].bias),
                          sub_buf_off[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS);
        H = (H+2-3)/2+1; W = (W+2-3)/2+1;
        metal_gemm_nn(ctx, E, P, WF(sub_conv[6].weight), SUB_CHANNELS, SUB_CHANNELS,
                      sub_buf_off[0], H*W, sub_buf_off[1]);
        metal_bias_relu_nchw_fp16(ctx, E, P, sub_buf_off[1], WF(sub_conv[6].bias), SUB_CHANNELS, H*W);

        T = H;
        metal_reshape_chw_to_hcw_fp16(ctx, E, P, sub_buf_off[1], sub_buf_off[0], SUB_CHANNELS, T, W);
        metal_gemm_nn_bias(ctx, E, P, sub_buf_off[0], T, SUB_CHANNELS*W,
                           WF(sub_out_w), D_MODEL, WF(sub_out_b), x_off);

        pt.end(enc);
        fprintf(stderr, "  subsampling:  %6.2f ms  (T=%d)\n", pt.gpu_ms, T);
        total_ms += pt.gpu_ms;
    }

    size_t pos_enc_T = pos_enc_off + (size_t)(T_max - T) * D_MODEL * sizeof(half_t);
    int pos_len = 2 * T - 1;

    // Per-block profiling with sub-phase breakdown
    double block_total = 0;
    double ff1_total = 0, mhsa_total = 0, conv_total = 0, ff2_total = 0;

    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = g_wo.blocks[blk];

        // FF1
        double ff1_ms, mhsa_ms, conv_ms, ff2_ms;
        {
            PhaseTimer pt{ctx};
            auto enc = pt.begin(); void* E = (__bridge void*)enc;
            metal_layer_norm_fp16(ctx, E, P, x_off, WF(blocks[blk].ff1_ln_w),
                                  WF(blocks[blk].ff1_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);
            metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].ff1_w1), D_FF, ff_mid_off);
            metal_silu_inplace_fp16(ctx, E, P, ff_mid_off, T * D_FF);
            metal_gemm_nn(ctx, E, P, ff_mid_off, T, D_FF, WF(blocks[blk].ff1_w2), D_MODEL, ff_out_off);
            metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, ff_out_off, 0.5f,
                WF(blocks[blk].mhsa_ln_w), WF(blocks[blk].mhsa_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);
            pt.end(enc);
            ff1_ms = pt.gpu_ms;
        }

        // MHSA
        {
            PhaseTimer pt{ctx};
            auto enc = pt.begin(); void* E = (__bridge void*)enc;
            metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, qkv_w_off[blk], 3*D_MODEL, qkv_off);
            metal_split_transpose_qkv_bias_fp16(ctx, E, P, qkv_off,
                WF(blocks[blk].pos_bias_u), WF(blocks[blk].pos_bias_v),
                q_u_off, q_v_buf_off, k_off, v_off, T, N_HEADS, HEAD_DIM);
            metal_gemm_nn(ctx, E, P, pos_enc_T, pos_len, D_MODEL,
                          WF(blocks[blk].pos_w), D_MODEL, pos_proj_off);
            metal_batched_gemm_nt_ex(ctx, E, P,
                q_v_buf_off, HEAD_DIM, (int64_t)T * HEAD_DIM,
                pos_proj_off, D_MODEL, (int64_t)HEAD_DIM,
                pos_scores_off, pos_len, (int64_t)T * pos_len,
                N_HEADS, T, pos_len, HEAD_DIM);
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            metal_batched_gemm_nt(ctx, E, P, q_u_off, k_off, scores_off,
                N_HEADS, T, T, HEAD_DIM,
                (int64_t)T*HEAD_DIM, (int64_t)T*HEAD_DIM, (int64_t)T*T);
            metal_fused_score_softmax_fp16(ctx, E, P, scores_off, pos_scores_off,
                                            scores_off, N_HEADS, T, scale);
            metal_batched_gemm_nn(ctx, E, P, scores_off, v_off, attn_out_off,
                N_HEADS, T, HEAD_DIM, T,
                (int64_t)T*T, (int64_t)T*HEAD_DIM, (int64_t)T*HEAD_DIM);
            metal_transpose_0213_fp16(ctx, E, P, attn_out_off, ff_out_off, N_HEADS, T, HEAD_DIM);
            metal_gemm_nn(ctx, E, P, ff_out_off, T, D_MODEL, WF(blocks[blk].out_w), D_MODEL, mhsa_out_off);
            metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, mhsa_out_off, 1.0f,
                WF(blocks[blk].conv_ln_w), WF(blocks[blk].conv_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);
            pt.end(enc);
            mhsa_ms = pt.gpu_ms;
        }

        // Conv module
        {
            PhaseTimer pt{ctx};
            auto enc = pt.begin(); void* E = (__bridge void*)enc;
            metal_gemm_nt(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].conv_pw1_w), D_CONV_PW, conv_mid_off);
            metal_glu_fp16(ctx, E, P, conv_mid_off, conv_glu_off, T, D_MODEL);
            metal_depthwise_conv1d_k9_silu_fp16(ctx, E, P, conv_glu_off,
                WF(blocks[blk].conv_dw_w), WF(blocks[blk].conv_dw_b), conv_dw_off, T, D_MODEL);
            metal_gemm_nt(ctx, E, P, conv_dw_off, T, D_MODEL, WF(blocks[blk].conv_pw2_w), D_MODEL, mhsa_out_off);
            metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, mhsa_out_off, 1.0f,
                WF(blocks[blk].ff2_ln_w), WF(blocks[blk].ff2_ln_b), ln_out_off, T, D_MODEL, NORM_EPS);
            pt.end(enc);
            conv_ms = pt.gpu_ms;
        }

        // FF2
        {
            PhaseTimer pt{ctx};
            auto enc = pt.begin(); void* E = (__bridge void*)enc;
            metal_gemm_nn(ctx, E, P, ln_out_off, T, D_MODEL, WF(blocks[blk].ff2_w1), D_FF, ff_mid_off);
            metal_silu_inplace_fp16(ctx, E, P, ff_mid_off, T * D_FF);
            metal_gemm_nn(ctx, E, P, ff_mid_off, T, D_FF, WF(blocks[blk].ff2_w2), D_MODEL, ff_out_off);
            metal_residual_add_layer_norm_fp16(ctx, E, P, x_off, ff_out_off, 0.5f,
                WF(blocks[blk].final_ln_w), WF(blocks[blk].final_ln_b), x_off, T, D_MODEL, NORM_EPS);
            pt.end(enc);
            ff2_ms = pt.gpu_ms;
        }

        double blk_ms = ff1_ms + mhsa_ms + conv_ms + ff2_ms;
        if (blk < 3 || blk == N_BLOCKS-1) {
            fprintf(stderr, "  block[%2d]:    %6.2f ms  (ff1=%.2f mhsa=%.2f conv=%.2f ff2=%.2f)\n",
                    blk, blk_ms, ff1_ms, mhsa_ms, conv_ms, ff2_ms);
        } else if (blk == 3) {
            fprintf(stderr, "  ... (blocks 3-%d similar) ...\n", N_BLOCKS-2);
        }
        block_total += blk_ms;
        ff1_total += ff1_ms; mhsa_total += mhsa_ms;
        conv_total += conv_ms; ff2_total += ff2_ms;
    }
    fprintf(stderr, "  blocks avg:   %6.2f ms  (ff1=%.2f mhsa=%.2f conv=%.2f ff2=%.2f)\n",
            block_total/N_BLOCKS, ff1_total/N_BLOCKS, mhsa_total/N_BLOCKS,
            conv_total/N_BLOCKS, ff2_total/N_BLOCKS);
    total_ms += block_total;

    // Encoder projection
    {
        PhaseTimer pt{ctx};
        auto enc = pt.begin(); void* E = (__bridge void*)enc;
        metal_gemm_nn_bias(ctx, E, P, x_off, T, D_MODEL,
                           WF(enc_proj_w), D_JOINT, WF(enc_proj_b), enc_proj_all_off);
        pt.end(enc);
        fprintf(stderr, "  enc_proj:     %6.2f ms\n", pt.gpu_ms);
        total_ms += pt.gpu_ms;
    }

    fprintf(stderr, "  ─────────────────────\n");
    fprintf(stderr, "  TOTAL GPU:    %6.2f ms\n\n", total_ms);

    return T;
}

#undef WF
