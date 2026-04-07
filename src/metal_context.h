// metal_context.h — Metal device, command queue, and pipeline state cache
//
// Pimpl pattern: the header is pure C++ (no Objective-C types) so it can be
// included from .cpp files. The Obj-C++ implementation lives in metal_context.mm.

#pragma once

#include <memory>
#include <string>

struct MetalContextImpl;  // opaque, defined in metal_context.mm

struct MetalContext {
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) noexcept;
    MetalContext& operator=(MetalContext&&) noexcept;

    // Load Metal shader source and compile into a library.
    // Can be called multiple times to add more shader sources.
    void load_shaders(const char* source, const char* label = "paraketto");

    // Allocate a shared (CPU+GPU) buffer. Returns an opaque handle.
    // On Apple Silicon unified memory, both CPU and GPU access the same physical
    // memory — no explicit copies needed.
    void* alloc_shared(size_t bytes);

    // Allocate a shared buffer backed by existing memory (zero-copy).
    // The caller must ensure `ptr` stays valid for the buffer's lifetime.
    // `ptr` must be page-aligned, `bytes` must be a multiple of page size.
    void* alloc_shared_nocopy(void* ptr, size_t bytes);

    // Free a buffer allocated by alloc_shared or alloc_shared_nocopy.
    void free_buffer(void* handle);

    // Get the CPU-accessible pointer for a shared buffer.
    void* buffer_contents(void* handle);

    // Get the maximum threadgroup memory size (bytes).
    size_t max_threadgroup_memory() const;

    std::unique_ptr<MetalContextImpl> impl;
};
