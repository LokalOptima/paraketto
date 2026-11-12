// common.h — Shared constants and CUDA helpers for Parakeet backends
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

namespace paraketto {

// What the library throws when it cannot go on: a weights or WAV file it
// cannot use, or a failed GPU call. what() is one line naming the cause.
// `cuda` is the code of the failed CUDA call (cudaErrorMemoryAllocation when
// the GPU has no room), or cudaSuccess when the failure was not a CUDA call.
// The library never ends the process and its destructors never throw.
struct Error : std::runtime_error {
    cudaError_t cuda;
    explicit Error(const std::string& message, cudaError_t code = cudaSuccess)
        : std::runtime_error(message), cuda(code) {}
};

inline std::string format_message(const char* format, va_list arguments) {
    va_list measured;
    va_copy(measured, arguments);
    const int length = vsnprintf(nullptr, 0, format, measured);
    va_end(measured);
    std::string message(length > 0 ? (size_t)length : 0, '\0');
    vsnprintf(&message[0], message.size() + 1, format, arguments);
    return message;
}

// Throw Error with a printf-style message. Kept out of line: the checks that
// call these sit in the decode loop.
[[noreturn]] __attribute__((cold, noinline, format(printf, 1, 2)))
inline void fail(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const std::string message = format_message(format, arguments);
    va_end(arguments);
    throw Error(message);
}

[[noreturn]] __attribute__((cold, noinline, format(printf, 2, 3)))
inline void fail(cudaError_t code, const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const std::string message = format_message(format, arguments);
    va_end(arguments);
    throw Error(message, code);
}

[[noreturn]] __attribute__((cold, noinline))
inline void fail_cuda(cudaError_t code, const char* file, int line) {
    fail(code, "CUDA error at %s:%d: %s", file, line, cudaGetErrorString(code));
}

} // namespace paraketto

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
            paraketto::fail_cuda(err, __FILE__, __LINE__);                     \
    } while (0)

// cuBLAS error checking (usable after including <cublas_v2.h>)
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        auto _cublas_stat = (call);                                            \
        if ((int)_cublas_stat != 0)                                            \
            paraketto::fail("cuBLAS error at %s:%d: %d",                       \
                            __FILE__, __LINE__, (int)_cublas_stat);             \
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
