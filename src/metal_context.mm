// metal_context.mm — Objective-C++ implementation of MetalContext
//
// Manages the Metal device, command queue, shader library, and pipeline cache.
// Shader source is compiled at runtime (like llama.cpp's ggml-metal backend).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_context.h"
#include "metal_context_impl.h"

#include <cstdio>

// ---------------------------------------------------------------------------
// MetalContext lifetime
// ---------------------------------------------------------------------------

MetalContext::MetalContext() : impl(std::make_unique<MetalContextImpl>()) {
    impl->device = MTLCreateSystemDefaultDevice();
    METAL_CHECK(impl->device != nil, "Metal is not supported on this device");

    impl->queue = [impl->device newCommandQueue];
    METAL_CHECK(impl->queue != nil, "Failed to create Metal command queue");

    fprintf(stderr, "[metal] device: %s\n",
            impl->device.name.UTF8String);
}

MetalContext::~MetalContext() = default;

MetalContext::MetalContext(MetalContext&&) noexcept = default;
MetalContext& MetalContext::operator=(MetalContext&&) noexcept = default;

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------

void MetalContext::load_shaders(const char* source, const char* label) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    opts.mathMode = MTLMathModeFast;

    NSString* src = [NSString stringWithUTF8String:source];
    id<MTLLibrary> lib = [impl->device newLibraryWithSource:src
                                                    options:opts
                                                      error:&error];
    if (!lib) {
        fprintf(stderr, "[metal] shader compile error (%s):\n%s\n",
                label, error.localizedDescription.UTF8String);
        std::exit(1);
    }

    // If we already have a library, we'd need to merge — for now, just replace.
    // In practice we'll compile all shaders from a single concatenated source.
    impl->library = lib;

    // Log available functions
    NSArray<NSString*>* names = [lib functionNames];
    fprintf(stderr, "[metal] compiled %lu kernel(s) from '%s'\n",
            (unsigned long)names.count, label);
}

// ---------------------------------------------------------------------------
// Buffer allocation
// ---------------------------------------------------------------------------

void* MetalContext::alloc_shared(size_t bytes) {
    id<MTLBuffer> buf = [impl->device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
    METAL_CHECK(buf != nil, "Failed to allocate Metal shared buffer");
    return (void*)CFBridgingRetain(buf);
}

void* MetalContext::alloc_shared_nocopy(void* ptr, size_t bytes) {
    id<MTLBuffer> buf = [impl->device
        newBufferWithBytesNoCopy:ptr
                         length:bytes
                        options:MTLResourceStorageModeShared
                    deallocator:nil];
    METAL_CHECK(buf != nil, "Failed to create Metal no-copy buffer");
    return (void*)CFBridgingRetain(buf);
}

void MetalContext::free_buffer(void* handle) {
    if (handle) {
        CFBridgingRelease(handle);
    }
}

void* MetalContext::buffer_contents(void* handle) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle;
    return buf.contents;
}

size_t MetalContext::max_threadgroup_memory() const {
    return impl->device.maxThreadgroupMemoryLength;
}
