// Force the no-algorithm path through the linker, without a runtime test hook.
// Run: make test-cublas-fallback
#include "gemm.h"
#include "common.h"
#include <cublasLt.h>
#include <cmath>
#include <vector>

static int heuristic_calls = 0;

extern "C" cublasStatus_t __wrap_cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int,
    cublasLtMatmulHeuristicResult_t*, int* count) {
    ++heuristic_calls;
    *count = 0;
    return CUBLAS_STATUS_SUCCESS;
}

int main() {
    using namespace paraketto;
    constexpr int M = 3, N = 16, K = 16;
    std::vector<half> x(M * K), w(K * N), wt(N * K), bias(N), actual(M * N);
    for (int i = 0; i < M * K; ++i) x[i] = __float2half((i % 5 - 2) * 0.25f);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            wt[n * K + k] = w[k * N + n] = __float2half(((k + n) % 7 - 3) * 0.25f);
    for (int n = 0; n < N; ++n) bias[n] = __float2half((n + 1) * 0.125f);

    half *dx, *dw, *dwt, *db, *dy;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dw, w.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dwt, wt.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&db, bias.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, actual.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dwt, wt.data(), wt.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, bias.data(), bias.size() * sizeof(half), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    gemm_init(stream);
    for (bool transpose : {false, true}) {
        for (bool with_bias : {false, true}) {
            // Repeat each call to cover reuse of a cached fallback plan.
            for (int repeat = 0; repeat < 2; ++repeat) {
                CUDA_CHECK(cudaMemsetAsync(dy, 0xff, actual.size() * sizeof(half), stream));
                if (transpose && with_bias) gemm_nt_bias(stream, dx, M, K, dwt, N, db, dy);
                else if (transpose) gemm_nt(stream, dx, M, K, dwt, N, dy);
                else if (with_bias) gemm_nn_bias(stream, dx, M, K, dw, N, db, dy);
                else gemm_nn(stream, dx, M, K, dw, N, dy);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(actual.data(), dy, actual.size() * sizeof(half), cudaMemcpyDeviceToHost));
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        float expected = with_bias ? __half2float(bias[n]) : 0.0f;
                        for (int k = 0; k < K; ++k)
                            expected += __half2float(x[m * K + k]) * __half2float(w[k * N + n]);
                        float value = __half2float(actual[m * N + n]);
                        // Dyadic inputs make the complete result exactly representable.
                        if (!std::isfinite(value) || value != expected) {
                            fprintf(stderr, "Fallback %s bias=%d [%d,%d]: expected %g, got %g\n",
                                    transpose ? "NT" : "NN", with_bias, m, n, expected, value);
                            return 1;
                        }
                    }
                }
            }
        }
    }
    gemm_free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (half* p : {dx, dw, dwt, db, dy}) CUDA_CHECK(cudaFree(p));
    if (heuristic_calls != 4) {
        fprintf(stderr, "Expected four forced fallback plans, got %d\n", heuristic_calls);
        return 1;
    }
    puts("cuBLAS fallback: NN/NT, with/without bias, and cached plans passed.");
}
