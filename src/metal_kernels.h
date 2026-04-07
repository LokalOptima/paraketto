// metal_kernels.h — Metal kernel launch wrappers for Parakeet conformer
//
// Mirrors kernels.h but for the Metal backend. All functions encode compute
// commands onto a MTLComputeCommandEncoder (passed as opaque void*).
// Buffer arguments are offsets into the pooled MTLBuffer.

#pragma once

#include <cstddef>
#include <cstdint>

struct MetalContext;

// Opaque handle to id<MTLComputeCommandEncoder>
using MetalEncoder = void*;
// Opaque handle to id<MTLBuffer>
using MetalBuffer = void*;

// Mel filterbank entry (matches kernels.h MelFBEntry)
struct MelFBEntry;

void metal_layer_norm_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                           size_t x_off, size_t gamma_off, size_t beta_off,
                           size_t y_off, int N, int D, float eps);

void metal_residual_add_layer_norm_fp16(MetalContext& ctx, MetalEncoder enc,
                                        MetalBuffer pool,
                                        size_t x_off, size_t delta_off, float alpha,
                                        size_t gamma_off, size_t beta_off,
                                        size_t ln_out_off, int N, int D, float eps);

void metal_silu_inplace_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                             size_t x_off, int n);

void metal_glu_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                    size_t x_off, size_t y_off, int N, int D);

void metal_add_relu_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                         size_t a_off, size_t b_off, size_t y_off, int n);

void metal_dual_argmax_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                            size_t logits_off, size_t out_off,
                            int vocab_size, int total);

void metal_depthwise_conv1d_k9_silu_fp16(MetalContext& ctx, MetalEncoder enc,
                                          MetalBuffer pool,
                                          size_t x_off, size_t w_off, size_t b_off,
                                          size_t y_off, int T, int C);

void metal_lstm_cell_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                          size_t gates_off, size_t c_prev_off,
                          size_t h_out_off, size_t c_out_off, int H);

void metal_embed_concat_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                             size_t table_off, int idx, size_t h_off,
                             size_t out_off, int D);

void metal_concat_vectors_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                               size_t a_off, size_t b_off, size_t out_off, int D);

void metal_cast_fp32_to_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                             size_t x_off, size_t y_off, int n);

void metal_transpose_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                          size_t x_off, size_t y_off, int M, int N);

void metal_residual_add_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                             size_t a_off, size_t b_off, size_t y_off,
                             int n, float alpha);

void metal_fused_score_softmax_fp16(MetalContext& ctx, MetalEncoder enc,
                                     MetalBuffer pool,
                                     size_t content_off, size_t pos_raw_off,
                                     size_t output_off,
                                     int heads, int T, float scale);

void metal_conv2d_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                       size_t input_off, size_t weight_off, size_t bias_off,
                       size_t output_off,
                       int C_in, int H_in, int W_in, int C_out,
                       int kH, int kW, int stride, int pad, int groups);

void metal_im2col_2d_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                          size_t input_off, size_t col_off,
                          int C_in, int H_in, int W_in,
                          int kH, int kW, int stride, int pad,
                          int H_out, int W_out);

void metal_bias_add_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                         size_t x_off, size_t bias_off, int rows, int cols);

void metal_bias_relu_nchw_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                               size_t x_off, size_t bias_off, int C, int spatial);

void metal_reshape_chw_to_hcw_fp16(MetalContext& ctx, MetalEncoder enc,
                                    MetalBuffer pool,
                                    size_t in_off, size_t out_off,
                                    int C, int H, int W);

void metal_generate_pos_encoding_gpu(MetalContext& ctx, MetalEncoder enc,
                                      MetalBuffer pool,
                                      size_t output_off, int T, int d_model);

void metal_transpose_0213_fp16(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                               size_t in_off, size_t out_off, int A, int B, int C);

void metal_split_transpose_qkv_bias_fp16(MetalContext& ctx, MetalEncoder enc,
                                          MetalBuffer pool,
                                          size_t in_off, size_t bias_u_off,
                                          size_t bias_v_off,
                                          size_t q_u_off, size_t q_v_off,
                                          size_t k_off, size_t v_off,
                                          int T, int heads, int head_dim);

// Mel pipeline
void metal_mel_init_filterbank(MetalContext& ctx, const MelFBEntry* entries, int count);

void metal_fft512_mel_log(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                          size_t frames_off, size_t mel_out_off, int n_frames);

void metal_mel_normalize(MetalContext& ctx, MetalEncoder enc, MetalBuffer pool,
                         size_t mel_in_off, size_t mel_out_off,
                         int n_frames, int n_valid);
