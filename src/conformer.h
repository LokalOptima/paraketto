// conformer.h — FP16 CudaModel (CUTLASS or cuBLAS backend via gemm.h)
//
// Shared buffers and the TDT decoder live in CudaModelBase (conformer_base.h).
// This variant adds pre-concatenated FP16 QKV weights and the FP16 encoder.
//
// Model constants, ModelConfig, and Weights are in model_defs.h (shared
// with the FP8 variant in conformer_fp8.h).

#ifndef CONFORMER_H_
#define CONFORMER_H_

#include "conformer_base.h"

namespace paraketto {

struct CudaModel : CudaModelBase {
    // Pre-concatenated QKV weights per block [D_MODEL, 3*D_MODEL]
    half* qkv_w[N_BLOCKS];

    void init(const Weights& weights, cudaStream_t s, int max_mel_frames);
    void free();

    int encode_gpu(int T_mel);
};

} // namespace paraketto

#endif  // CONFORMER_H_
