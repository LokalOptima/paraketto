// cuda_stubs.cpp — Minimal stubs for CUDA runtime registration symbols.
// nvcc-compiled .cu object files reference these at startup, but our
// cudaless code never actually launches device code through the runtime.
// Linking these stubs avoids pulling in libcudart.so.

#include <cstddef>

extern "C" {

void** __cudaRegisterFatBinary(void*) {
    static void* v = nullptr;
    return &v;
}

void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}

void __cudaRegisterVar(void**, char*, char*, const char*,
                       int, size_t, int, int) {}

void __cudaRegisterFunction(void**, const char*, char*, const char*,
                            int, void*, void*, void*, void*, int*) {}

}
