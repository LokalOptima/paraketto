// common_metal.h — Shared constants and Metal helpers for Parakeet backends
#pragma once

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define METAL_CHECK(expr, msg)                                                 \
    do {                                                                        \
        if (!(expr)) {                                                          \
            fprintf(stderr, "Metal error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, (msg));                                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constants — match NeMo Parakeet TDT 0.6B V2 preprocessor
// (duplicated from common.h to avoid CUDA dependency)
// ---------------------------------------------------------------------------

static constexpr int MAX_MEL_FRAMES = 16000 * 120 / 160;  // 120s max audio at 16kHz
static constexpr int N_FFT = 512;
static constexpr int HOP = 160;
static constexpr int N_MELS = 128;
static constexpr int N_FREQ = N_FFT / 2 + 1;  // 257
static constexpr float PREEMPH = 0.97f;
static constexpr float LOG_EPS = 5.9604645e-08f;
static constexpr float NORM_EPS = 1e-05f;
