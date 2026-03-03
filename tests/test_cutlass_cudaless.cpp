// test_cutlass_cudaless.cpp — Test CUTLASS GEMM via cudaless ioctls
//
// Tests 4 main variants: gemm_nn, gemm_nt, batched_nn, batched_nt
// Each test: generate random FP16 matrices, launch via gpu.h, compare to CPU reference.
// Tolerance: relative 2% for FP16 tensor core accumulation.

#include "cutlass_cudaless.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>

// =========================================================================
// FP16 helpers
// =========================================================================

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | frac;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { float f; uint32_t v = sign; memcpy(&f, &v, 4); return f; }
        // Denorm
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        exp++; frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t v = sign | 0x7F800000 | (frac << 13);
        float f; memcpy(&f, &v, 4); return f;
    }
    uint32_t v = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; memcpy(&f, &v, 4);
    return f;
}

static void fill_random_fp16(uint16_t* data, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f; // [-1, 1]
        data[i] = fp32_to_fp16(val);
    }
}

// =========================================================================
// CPU reference GEMM
// =========================================================================

// C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n]  (row-major)
static void cpu_gemm_nn(const uint16_t* A, const uint16_t* B, uint16_t* C,
                         int m, int n, int k, float alpha, float beta) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += fp16_to_fp32(A[i * k + p]) * fp16_to_fp32(B[p * n + j]);
            float old = fp16_to_fp32(C[i * n + j]);
            C[i * n + j] = fp32_to_fp16(alpha * sum + beta * old);
        }
    }
}

// C[m,n] = alpha * A[m,k] @ B[n,k]^T + beta * C[m,n]  (row-major)
static void cpu_gemm_nt(const uint16_t* A, const uint16_t* B, uint16_t* C,
                         int m, int n, int k, float alpha, float beta) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += fp16_to_fp32(A[i * k + p]) * fp16_to_fp32(B[j * k + p]);
            float old = fp16_to_fp32(C[i * n + j]);
            C[i * n + j] = fp32_to_fp16(alpha * sum + beta * old);
        }
    }
}

// =========================================================================
// Comparison helper
// =========================================================================

static bool compare_fp16(const uint16_t* gpu, const uint16_t* cpu, int count,
                          float rtol, float atol, const char* name) {
    int errors = 0;
    float max_rel = 0;
    for (int i = 0; i < count; i++) {
        float g = fp16_to_fp32(gpu[i]);
        float c = fp16_to_fp32(cpu[i]);
        float diff = fabsf(g - c);
        float rel = (fabsf(c) > 1e-6f) ? diff / fabsf(c) : diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > atol && rel > rtol) {
            if (errors < 5)
                printf("    MISMATCH [%d]: gpu=%.6f cpu=%.6f diff=%.6f rel=%.4f\n",
                       i, g, c, diff, rel);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  %s: PASS (max_rel=%.4f)\n", name, max_rel);
        return true;
    } else {
        printf("  %s: FAIL (%d/%d errors, max_rel=%.4f)\n", name, errors, count, max_rel);
        return false;
    }
}

// =========================================================================
// Test functions
// =========================================================================

static bool test_gemm_nn(CutlassCudaless& cl) {
    const int M = 64, N = 128, K = 64;

    // Allocate GPU memory
    auto alloc_A = cl.gpu->gpu_malloc(M * K * 2);
    auto alloc_B = cl.gpu->gpu_malloc(K * N * 2);
    auto alloc_C = cl.gpu->gpu_malloc(M * N * 2);

    // Generate random data
    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, M * K, 42);
    fill_random_fp16(h_B, K * N, 123);
    memset(h_C, 0, M * N * 2);
    __sync_synchronize();

    // CPU reference
    uint16_t* cpu_C = new uint16_t[M * N]();
    cpu_gemm_nn(h_A, h_B, cpu_C, M, N, K, 1.0f, 0.0f);

    // GPU launch
    cl.gpu->begin_commands();
    cl.gemm_nn(alloc_A.gpu_addr, M, K, alloc_B.gpu_addr, N, alloc_C.gpu_addr);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, M * N, 0.02f, 0.01f, "gemm_nn 64x128x64");
    delete[] cpu_C;
    return ok;
}

static bool test_gemm_nt(CutlassCudaless& cl) {
    const int M = 64, N = 128, K = 64;

    auto alloc_A = cl.gpu->gpu_malloc(M * K * 2);
    auto alloc_B = cl.gpu->gpu_malloc(N * K * 2);  // B is [N, K] for NT
    auto alloc_C = cl.gpu->gpu_malloc(M * N * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, M * K, 77);
    fill_random_fp16(h_B, N * K, 88);
    memset(h_C, 0, M * N * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[M * N]();
    cpu_gemm_nt(h_A, h_B, cpu_C, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.gemm_nt(alloc_A.gpu_addr, M, K, alloc_B.gpu_addr, N, alloc_C.gpu_addr);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, M * N, 0.02f, 0.01f, "gemm_nt 64x128x64");
    delete[] cpu_C;
    return ok;
}

static bool test_batched_nn(CutlassCudaless& cl) {
    const int batch = 4, M = 64, N = 64, K = 64;
    long long sA = M * K, sB = K * N, sC = M * N;

    auto alloc_A = cl.gpu->gpu_malloc(batch * sA * 2);
    auto alloc_B = cl.gpu->gpu_malloc(batch * sB * 2);
    auto alloc_C = cl.gpu->gpu_malloc(batch * sC * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, batch * M * K, 111);
    fill_random_fp16(h_B, batch * K * N, 222);
    memset(h_C, 0, batch * sC * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[batch * M * N]();
    for (int b = 0; b < batch; b++)
        cpu_gemm_nn(h_A + b * sA, h_B + b * sB, cpu_C + b * sC, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.batched_gemm_nn(alloc_A.gpu_addr, alloc_B.gpu_addr, alloc_C.gpu_addr,
                       batch, M, N, K, sA, sB, sC);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, batch * M * N, 0.02f, 0.01f,
                           "batched_nn 4x64x64x64");
    delete[] cpu_C;
    return ok;
}

static bool test_batched_nt(CutlassCudaless& cl) {
    const int batch = 4, M = 64, N = 64, K = 64;
    long long sA = M * K, sB = N * K, sC = M * N;

    auto alloc_A = cl.gpu->gpu_malloc(batch * sA * 2);
    auto alloc_B = cl.gpu->gpu_malloc(batch * sB * 2);
    auto alloc_C = cl.gpu->gpu_malloc(batch * sC * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, batch * M * K, 333);
    fill_random_fp16(h_B, batch * N * K, 444);
    memset(h_C, 0, batch * sC * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[batch * M * N]();
    for (int b = 0; b < batch; b++)
        cpu_gemm_nt(h_A + b * sA, h_B + b * sB, cpu_C + b * sC, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.batched_gemm_nt(alloc_A.gpu_addr, alloc_B.gpu_addr, alloc_C.gpu_addr,
                       batch, M, N, K, sA, sB, sC);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, batch * M * N, 0.02f, 0.01f,
                           "batched_nt 4x64x64x64");
    delete[] cpu_C;
    return ok;
}

// =========================================================================
// Main
// =========================================================================

int main() {
    printf("=== CUTLASS Cudaless Test ===\n\n");

    GPU gpu;
    gpu.init();

    CutlassCudaless cl;
    if (!cl.init(gpu, "cutlass_gemm.cubin")) {
        fprintf(stderr, "Failed to init CutlassCudaless\n");
        return 1;
    }
    printf("Loaded %zu CUTLASS kernels\n\n", cl.cubin.kernels.size());

    int pass = 0, total = 0;

    total++; if (test_gemm_nn(cl)) pass++;
    total++; if (test_gemm_nt(cl)) pass++;
    total++; if (test_batched_nn(cl)) pass++;
    total++; if (test_batched_nt(cl)) pass++;

    printf("\n=== Results: %d/%d PASS ===\n", pass, total);
    return (pass == total) ? 0 : 1;
}
