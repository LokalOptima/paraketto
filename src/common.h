// common.h — Shared constants and CUDA helpers for Parakeet backends
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// cuBLAS error checking (usable after including <cublas_v2.h>)
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        auto _cublas_stat = (call);                                            \
        if ((int)_cublas_stat != 0) {                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, (int)_cublas_stat);                     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants — match NeMo Parakeet TDT 0.6B V2 preprocessor
// ---------------------------------------------------------------------------

namespace paraketto {

static constexpr int MAX_MEL_FRAMES = 16000 * 120 / 160;  // 120s max audio at 16kHz
static constexpr int N_FFT = 512;
static constexpr int HOP = 160;
static constexpr int N_MELS = 128;
static constexpr int N_FREQ = N_FFT / 2 + 1;  // 257
static constexpr float PREEMPH = 0.97f;
static constexpr float LOG_EPS = 5.9604645e-08f;
static constexpr float NORM_EPS = 1e-05f;

} // namespace paraketto
