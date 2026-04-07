// metal_kernels.mm — Metal kernel dispatch for Parakeet conformer
//
// Each function encodes a compute command onto the given encoder.
// Buffer arguments are offsets into the pooled MTLBuffer.

#import <Metal/Metal.h>

#include "metal_kernels.h"
#include "metal_context.h"
#include "metal_context_impl.h"
#include "common_metal.h"

// Define MelFBEntry here (matches kernels.h and metal/kernels.metal)
struct MelFBEntry {
    uint16_t freq;
    uint16_t mel;
    float    weight;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline id<MTLComputeCommandEncoder> enc(MetalEncoder e) {
    return (__bridge id<MTLComputeCommandEncoder>)e;
}

static inline id<MTLBuffer> buf(MetalBuffer b) {
    return (__bridge id<MTLBuffer>)b;
}

static void dispatch_1d(MetalContext& ctx, MetalEncoder encoder,
                        const char* kernel_name,
                        id<MTLBuffer> pool, uint total_threads,
                        void (^set_args)(id<MTLComputeCommandEncoder>)) {
    auto pso = ctx.impl->get_pipeline(kernel_name);
    auto e = enc(encoder);
    [e setComputePipelineState:pso];
    set_args(e);

    NSUInteger tg_size = MIN(total_threads,
                             pso.maxTotalThreadsPerThreadgroup);
    // Round down to multiple of threadExecutionWidth
    NSUInteger exec_width = pso.threadExecutionWidth;
    if (tg_size > exec_width) {
        tg_size = (tg_size / exec_width) * exec_width;
    }
    MTLSize grid = MTLSizeMake(total_threads, 1, 1);
    MTLSize group = MTLSizeMake(tg_size, 1, 1);
    [e dispatchThreads:grid threadsPerThreadgroup:group];
}

static void dispatch_groups(MetalContext& ctx, MetalEncoder encoder,
                            const char* kernel_name,
                            MTLSize grid_groups, MTLSize group_size,
                            void (^set_args)(id<MTLComputeCommandEncoder>)) {
    auto pso = ctx.impl->get_pipeline(kernel_name);
    auto e = enc(encoder);
    [e setComputePipelineState:pso];
    set_args(e);
    [e dispatchThreadgroups:grid_groups threadsPerThreadgroup:group_size];
}

// Mel filterbank stored for FFT kernel
static id<MTLBuffer> g_mel_fb_buf = nil;
static uint g_mel_fb_count = 0;

// ---------------------------------------------------------------------------
// Kernel dispatchers
// ---------------------------------------------------------------------------

void metal_layer_norm_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                           size_t x_off, size_t gamma_off, size_t beta_off,
                           size_t y_off, int N, int D, float eps) {
    uint threads = (D <= 1024) ? (((D + 31) / 32) * 32) : 1024;
    if (threads < 32) threads = 32;

    dispatch_groups(ctx, encoder, "layer_norm_kernel",
                    MTLSizeMake(N, 1, 1), MTLSizeMake(threads, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off     atIndex:0];
        [e setBuffer:buf(pool) offset:gamma_off  atIndex:1];
        [e setBuffer:buf(pool) offset:beta_off   atIndex:2];
        [e setBuffer:buf(pool) offset:y_off      atIndex:3];
        uint n = N, d = D;
        [e setBytes:&n   length:sizeof(n)   atIndex:4];
        [e setBytes:&d   length:sizeof(d)   atIndex:5];
        [e setBytes:&eps length:sizeof(eps) atIndex:6];
    });
}

void metal_residual_add_layer_norm_fp16(MetalContext& ctx, MetalEncoder encoder,
                                        MetalBuffer pool,
                                        size_t x_off, size_t delta_off, float alpha,
                                        size_t gamma_off, size_t beta_off,
                                        size_t ln_out_off, int N, int D, float eps) {
    uint threads = (D <= 1024) ? (((D + 31) / 32) * 32) : 1024;
    if (threads < 32) threads = 32;

    dispatch_groups(ctx, encoder, "residual_add_layer_norm_kernel",
                    MTLSizeMake(N, 1, 1), MTLSizeMake(threads, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off       atIndex:0];
        [e setBuffer:buf(pool) offset:delta_off    atIndex:1];
        [e setBytes:&alpha length:sizeof(alpha)    atIndex:2];
        [e setBuffer:buf(pool) offset:gamma_off    atIndex:3];
        [e setBuffer:buf(pool) offset:beta_off     atIndex:4];
        [e setBuffer:buf(pool) offset:ln_out_off   atIndex:5];
        uint n = N, d = D;
        [e setBytes:&n   length:sizeof(n)   atIndex:6];
        [e setBytes:&d   length:sizeof(d)   atIndex:7];
        [e setBytes:&eps length:sizeof(eps) atIndex:8];
    });
}

void metal_silu_inplace_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                             size_t x_off, int n) {
    dispatch_1d(ctx, encoder, "silu_inplace_kernel", buf(pool), n,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off atIndex:0];
        uint nn = n;
        [e setBytes:&nn length:sizeof(nn) atIndex:1];
    });
}

void metal_glu_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                    size_t x_off, size_t y_off, int N, int D) {
    dispatch_1d(ctx, encoder, "glu_kernel", buf(pool), N * D,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off atIndex:0];
        [e setBuffer:buf(pool) offset:y_off atIndex:1];
        uint n = N, d = D;
        [e setBytes:&n length:sizeof(n) atIndex:2];
        [e setBytes:&d length:sizeof(d) atIndex:3];
    });
}

void metal_add_relu_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                         size_t a_off, size_t b_off, size_t y_off, int n) {
    dispatch_1d(ctx, encoder, "add_relu_kernel", buf(pool), n,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:a_off atIndex:0];
        [e setBuffer:buf(pool) offset:b_off atIndex:1];
        [e setBuffer:buf(pool) offset:y_off atIndex:2];
        uint nn = n;
        [e setBytes:&nn length:sizeof(nn) atIndex:3];
    });
}

void metal_dual_argmax_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                            size_t logits_off, size_t out_off,
                            int vocab_size, int total) {
    dispatch_groups(ctx, encoder, "dual_argmax_kernel",
                    MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:logits_off atIndex:0];
        [e setBuffer:buf(pool) offset:out_off    atIndex:1];
        uint vs = vocab_size, t = total;
        [e setBytes:&vs length:sizeof(vs) atIndex:2];
        [e setBytes:&t  length:sizeof(t)  atIndex:3];
    });
}

void metal_depthwise_conv1d_k9_silu_fp16(MetalContext& ctx, MetalEncoder encoder,
                                          MetalBuffer pool,
                                          size_t x_off, size_t w_off, size_t b_off,
                                          size_t y_off, int T, int C) {
    dispatch_1d(ctx, encoder, "depthwise_conv1d_k9_silu_kernel", buf(pool), T * C,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off atIndex:0];
        [e setBuffer:buf(pool) offset:w_off atIndex:1];
        [e setBuffer:buf(pool) offset:b_off atIndex:2];
        [e setBuffer:buf(pool) offset:y_off atIndex:3];
        uint t = T, c = C;
        [e setBytes:&t length:sizeof(t) atIndex:4];
        [e setBytes:&c length:sizeof(c) atIndex:5];
    });
}

void metal_lstm_cell_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                          size_t gates_off, size_t c_prev_off,
                          size_t h_out_off, size_t c_out_off, int H) {
    dispatch_1d(ctx, encoder, "lstm_cell_kernel", buf(pool), H,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:gates_off  atIndex:0];
        [e setBuffer:buf(pool) offset:c_prev_off atIndex:1];
        [e setBuffer:buf(pool) offset:h_out_off  atIndex:2];
        [e setBuffer:buf(pool) offset:c_out_off  atIndex:3];
        uint h = H;
        [e setBytes:&h length:sizeof(h) atIndex:4];
    });
}

void metal_embed_concat_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                             size_t table_off, int idx, size_t h_off,
                             size_t out_off, int D) {
    dispatch_1d(ctx, encoder, "embed_concat_kernel", buf(pool), 2 * D,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:table_off atIndex:0];
        uint i = idx;
        [e setBytes:&i length:sizeof(i) atIndex:1];
        [e setBuffer:buf(pool) offset:h_off   atIndex:2];
        [e setBuffer:buf(pool) offset:out_off atIndex:3];
        uint d = D;
        [e setBytes:&d length:sizeof(d) atIndex:4];
    });
}

void metal_concat_vectors_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                               size_t a_off, size_t b_off, size_t out_off, int D) {
    dispatch_1d(ctx, encoder, "concat_vectors_kernel", buf(pool), 2 * D,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:a_off   atIndex:0];
        [e setBuffer:buf(pool) offset:b_off   atIndex:1];
        [e setBuffer:buf(pool) offset:out_off atIndex:2];
        uint d = D;
        [e setBytes:&d length:sizeof(d) atIndex:3];
    });
}

void metal_cast_fp32_to_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                             size_t x_off, size_t y_off, int n) {
    dispatch_1d(ctx, encoder, "cast_fp32_to_fp16_kernel", buf(pool), n,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off atIndex:0];
        [e setBuffer:buf(pool) offset:y_off atIndex:1];
        uint nn = n;
        [e setBytes:&nn length:sizeof(nn) atIndex:2];
    });
}

void metal_transpose_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                          size_t x_off, size_t y_off, int M, int N) {
    uint grid_x = (N + 31) / 32;
    uint grid_y = (M + 31) / 32;

    dispatch_groups(ctx, encoder, "transpose_kernel",
                    MTLSizeMake(grid_x, grid_y, 1), MTLSizeMake(32, 8, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off atIndex:0];
        [e setBuffer:buf(pool) offset:y_off atIndex:1];
        uint m = M, n = N;
        [e setBytes:&m length:sizeof(m) atIndex:2];
        [e setBytes:&n length:sizeof(n) atIndex:3];
    });
}

void metal_residual_add_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                             size_t a_off, size_t b_off, size_t y_off,
                             int n, float alpha) {
    dispatch_1d(ctx, encoder, "residual_add_kernel", buf(pool), n,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:a_off atIndex:0];
        [e setBuffer:buf(pool) offset:b_off atIndex:1];
        [e setBuffer:buf(pool) offset:y_off atIndex:2];
        uint nn = n;
        [e setBytes:&nn    length:sizeof(nn)    atIndex:3];
        [e setBytes:&alpha length:sizeof(alpha) atIndex:4];
    });
}

void metal_fused_score_softmax_fp16(MetalContext& ctx, MetalEncoder encoder,
                                     MetalBuffer pool,
                                     size_t content_off, size_t pos_raw_off,
                                     size_t output_off,
                                     int heads, int T, float scale) {
    uint n_groups = heads * T;
    uint threads = 256;

    dispatch_groups(ctx, encoder, "fused_score_softmax_kernel",
                    MTLSizeMake(n_groups, 1, 1), MTLSizeMake(threads, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:content_off atIndex:0];
        [e setBuffer:buf(pool) offset:pos_raw_off atIndex:1];
        [e setBuffer:buf(pool) offset:output_off  atIndex:2];
        uint t = T;
        [e setBytes:&t     length:sizeof(t)     atIndex:3];
        [e setBytes:&scale length:sizeof(scale) atIndex:4];
    });
}

void metal_conv2d_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                       size_t input_off, size_t weight_off, size_t bias_off,
                       size_t output_off,
                       int C_in, int H_in, int W_in, int C_out,
                       int kH, int kW, int stride, int pad, int groups) {
    int H_out = (H_in + 2 * pad - kH) / stride + 1;
    int W_out = (W_in + 2 * pad - kW) / stride + 1;
    uint total = C_out * H_out * W_out;
    int c_per_group = C_in / groups;

    dispatch_1d(ctx, encoder, "conv2d_kernel", buf(pool), total,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:input_off  atIndex:0];
        [e setBuffer:buf(pool) offset:weight_off atIndex:1];
        size_t actual_bias_off = (bias_off != SIZE_MAX) ? bias_off : 0;
        [e setBuffer:buf(pool) offset:actual_bias_off atIndex:2];
        [e setBuffer:buf(pool) offset:output_off atIndex:3];
        uint ci = C_in, hi = H_in, wi = W_in, co = C_out;
        uint ho = H_out, wo = W_out;
        uint kkh = kH, kkw = kW, st = stride, pd = pad;
        uint gr = groups, cpg = c_per_group;
        [e setBytes:&ci  length:sizeof(ci)  atIndex:4];
        [e setBytes:&hi  length:sizeof(hi)  atIndex:5];
        [e setBytes:&wi  length:sizeof(wi)  atIndex:6];
        [e setBytes:&co  length:sizeof(co)  atIndex:7];
        [e setBytes:&ho  length:sizeof(ho)  atIndex:8];
        [e setBytes:&wo  length:sizeof(wo)  atIndex:9];
        [e setBytes:&kkh length:sizeof(kkh) atIndex:10];
        [e setBytes:&kkw length:sizeof(kkw) atIndex:11];
        [e setBytes:&st  length:sizeof(st)  atIndex:12];
        [e setBytes:&pd  length:sizeof(pd)  atIndex:13];
        [e setBytes:&gr  length:sizeof(gr)  atIndex:14];
        [e setBytes:&cpg length:sizeof(cpg) atIndex:15];
        uint hb = (bias_off != SIZE_MAX) ? 1 : 0;
        [e setBytes:&hb  length:sizeof(hb)  atIndex:16];
    });
}

void metal_im2col_2d_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                          size_t input_off, size_t col_off,
                          int C_in, int H_in, int W_in,
                          int kH, int kW, int stride, int pad,
                          int H_out, int W_out) {
    uint total = C_in * kH * kW * H_out * W_out;

    dispatch_1d(ctx, encoder, "im2col_2d_kernel", buf(pool), total,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:input_off atIndex:0];
        [e setBuffer:buf(pool) offset:col_off   atIndex:1];
        uint ci = C_in, hi = H_in, wi = W_in;
        uint kkh = kH, kkw = kW, st = stride, pd = pad;
        uint ho = H_out, wo = W_out;
        [e setBytes:&ci  length:sizeof(ci)  atIndex:2];
        [e setBytes:&hi  length:sizeof(hi)  atIndex:3];
        [e setBytes:&wi  length:sizeof(wi)  atIndex:4];
        [e setBytes:&kkh length:sizeof(kkh) atIndex:5];
        [e setBytes:&kkw length:sizeof(kkw) atIndex:6];
        [e setBytes:&st  length:sizeof(st)  atIndex:7];
        [e setBytes:&pd  length:sizeof(pd)  atIndex:8];
        [e setBytes:&ho  length:sizeof(ho)  atIndex:9];
        [e setBytes:&wo  length:sizeof(wo)  atIndex:10];
    });
}

void metal_bias_add_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                         size_t x_off, size_t bias_off, int rows, int cols) {
    dispatch_1d(ctx, encoder, "bias_add_kernel", buf(pool), rows * cols,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off    atIndex:0];
        [e setBuffer:buf(pool) offset:bias_off atIndex:1];
        uint r = rows, c = cols;
        [e setBytes:&r length:sizeof(r) atIndex:2];
        [e setBytes:&c length:sizeof(c) atIndex:3];
    });
}

void metal_bias_relu_nchw_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                               size_t x_off, size_t bias_off, int C, int spatial) {
    dispatch_1d(ctx, encoder, "bias_relu_nchw_kernel", buf(pool), C * spatial,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:x_off    atIndex:0];
        [e setBuffer:buf(pool) offset:bias_off atIndex:1];
        uint cc = C, sp = spatial;
        [e setBytes:&cc length:sizeof(cc) atIndex:2];
        [e setBytes:&sp length:sizeof(sp) atIndex:3];
    });
}

void metal_reshape_chw_to_hcw_fp16(MetalContext& ctx, MetalEncoder encoder,
                                    MetalBuffer pool,
                                    size_t in_off, size_t out_off,
                                    int C, int H, int W) {
    dispatch_1d(ctx, encoder, "reshape_chw_to_hcw_kernel", buf(pool), C * H * W,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:in_off  atIndex:0];
        [e setBuffer:buf(pool) offset:out_off atIndex:1];
        uint cc = C, hh = H, ww = W;
        [e setBytes:&cc length:sizeof(cc) atIndex:2];
        [e setBytes:&hh length:sizeof(hh) atIndex:3];
        [e setBytes:&ww length:sizeof(ww) atIndex:4];
    });
}

void metal_generate_pos_encoding_gpu(MetalContext& ctx, MetalEncoder encoder,
                                      MetalBuffer pool,
                                      size_t output_off, int T, int d_model) {
    uint total = (2 * T - 1) * (d_model / 2);
    dispatch_1d(ctx, encoder, "generate_pos_encoding_kernel", buf(pool), total,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:output_off atIndex:0];
        uint t = T, d = d_model;
        [e setBytes:&t length:sizeof(t) atIndex:1];
        [e setBytes:&d length:sizeof(d) atIndex:2];
    });
}

void metal_transpose_0213_fp16(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                               size_t in_off, size_t out_off, int A, int B, int C) {
    dispatch_1d(ctx, encoder, "transpose_0213_kernel", buf(pool), A * B * C,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:in_off  atIndex:0];
        [e setBuffer:buf(pool) offset:out_off atIndex:1];
        uint a = A, b = B, c = C;
        [e setBytes:&a length:sizeof(a) atIndex:2];
        [e setBytes:&b length:sizeof(b) atIndex:3];
        [e setBytes:&c length:sizeof(c) atIndex:4];
    });
}

void metal_split_transpose_qkv_bias_fp16(MetalContext& ctx, MetalEncoder encoder,
                                          MetalBuffer pool,
                                          size_t in_off, size_t bias_u_off,
                                          size_t bias_v_off,
                                          size_t q_u_off, size_t q_v_off,
                                          size_t k_off, size_t v_off,
                                          int T, int heads, int head_dim) {
    uint D = heads * head_dim;
    uint total = T * 3 * D;

    dispatch_1d(ctx, encoder, "split_transpose_qkv_bias_kernel", buf(pool), total,
                ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:in_off     atIndex:0];
        [e setBuffer:buf(pool) offset:bias_u_off atIndex:1];
        [e setBuffer:buf(pool) offset:bias_v_off atIndex:2];
        [e setBuffer:buf(pool) offset:q_u_off    atIndex:3];
        [e setBuffer:buf(pool) offset:q_v_off    atIndex:4];
        [e setBuffer:buf(pool) offset:k_off      atIndex:5];
        [e setBuffer:buf(pool) offset:v_off      atIndex:6];
        uint t = T, h = heads, hd = head_dim;
        [e setBytes:&t  length:sizeof(t)  atIndex:7];
        [e setBytes:&h  length:sizeof(h)  atIndex:8];
        [e setBytes:&hd length:sizeof(hd) atIndex:9];
    });
}

// ---------------------------------------------------------------------------
// Mel pipeline
// ---------------------------------------------------------------------------

void metal_mel_init_filterbank(MetalContext& ctx, const MelFBEntry* entries, int count) {
    g_mel_fb_count = count;
    // Allocate directly via Metal device
    g_mel_fb_buf = [ctx.impl->device newBufferWithBytes:entries
                                                 length:count * sizeof(MelFBEntry)
                                                options:MTLResourceStorageModeShared];
    METAL_CHECK(g_mel_fb_buf != nil, "Failed to allocate mel filterbank buffer");
}

void metal_fft512_mel_log(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                          size_t frames_off, size_t mel_out_off, int n_frames) {
    uint n_mels = N_MELS;
    float log_eps_val = LOG_EPS;

    dispatch_groups(ctx, encoder, "fft512_mel_log_kernel",
                    MTLSizeMake(n_frames, 1, 1), MTLSizeMake(256, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool)  offset:frames_off  atIndex:0];
        [e setBuffer:buf(pool)  offset:mel_out_off atIndex:1];
        [e setBuffer:g_mel_fb_buf offset:0         atIndex:2];
        uint nf = n_frames;
        uint nme = g_mel_fb_count;
        [e setBytes:&nf        length:sizeof(nf)        atIndex:3];
        [e setBytes:&nme       length:sizeof(nme)       atIndex:4];
        [e setBytes:&n_mels    length:sizeof(n_mels)    atIndex:5];
        [e setBytes:&log_eps_val length:sizeof(log_eps_val) atIndex:6];
    });
}

void metal_mel_normalize(MetalContext& ctx, MetalEncoder encoder, MetalBuffer pool,
                         size_t mel_in_off, size_t mel_out_off,
                         int n_frames, int n_valid) {
    dispatch_groups(ctx, encoder, "mel_normalize_kernel",
                    MTLSizeMake(128, 1, 1), MTLSizeMake(256, 1, 1),
                    ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:buf(pool) offset:mel_in_off  atIndex:0];
        [e setBuffer:buf(pool) offset:mel_out_off atIndex:1];
        uint nf = n_frames, nv = n_valid;
        [e setBytes:&nf length:sizeof(nf) atIndex:2];
        [e setBytes:&nv length:sizeof(nv) atIndex:3];
    });
}
