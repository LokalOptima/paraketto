// metal_context_impl.h — Internal header for MetalContextImpl (Obj-C++ only)
//
// This header is ONLY included from .mm files that need direct access to
// Metal objects. It must NOT be included from .cpp or .h files.

#pragma once

#import <Metal/Metal.h>
#include <unordered_map>
#include <string>

#include "common_metal.h"

struct MetalContextImpl {
    id<MTLDevice>        device  = nil;
    id<MTLCommandQueue>  queue   = nil;
    id<MTLLibrary>       library = nil;

    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    id<MTLComputePipelineState> get_pipeline(const char* name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        NSError* error = nil;
        id<MTLFunction> fn = [library newFunctionWithName:
                              [NSString stringWithUTF8String:name]];
        METAL_CHECK(fn != nil, name);

        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&error];
        METAL_CHECK(pso != nil,
                    error.localizedDescription.UTF8String);

        pipelines[name] = pso;
        return pso;
    }
};
