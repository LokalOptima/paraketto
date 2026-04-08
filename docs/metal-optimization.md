# Metal GEMM Optimization Log

## Baseline (pre-optimization)

- M1 Max, 10.4s audio clip, ~131 encoder frames after 8x subsampling
- **40x RTFx** (encode ~167ms, decode ~57ms, mel ~11ms)
- GPU encode time: ~125ms, of which 93% in 24 conformer blocks
- Per-block avg: 4.83ms (ff1=1.53 mhsa=1.24 conv=0.61 ff2=1.47)
- 11 GEMMs per block dominate compute

## Profiling infrastructure

Added `--profile` flag to `paraketto.metal` that splits `encode_gpu` into
separate command buffers per phase, measuring `gpuEndTime - gpuStartTime` on each.
Reports per-block breakdown (ff1, mhsa, conv, ff2) and total GPU time.

Added `tests/test_metal_correctness.sh` with 16 LibriSpeech WAVs (diverse
speakers, 2-20s durations) with golden reference transcriptions. Every
kernel change is validated against all 16: `make paraketto.metal &&
./tests/test_metal_correctness.sh`.

## Approaches tried (and why they failed)

### 1. Store path: all simdgroups write (no improvement)

Changed `store_accumulators` so all 4 simdgroups write their own 32x16
sub-tile in parallel, instead of only simdgroup 0 writing the full 64x32 tile.

**Result:** No measurable change. The store happens once per threadgroup after
32 K-iterations of compute — it's ~3% of GEMM time.

### 2. Threadgroup memory bank conflict padding (no improvement)

Padded sa/sb strides from NK=32 to NK+1=33 to break bank conflict patterns
(Apple Silicon has 32 banks of 2 bytes).

**Result:** Within noise. `simdgroup_load` reads 8 consecutive elements,
not strided across banks, so no conflicts to fix.

### 3. Fused SiLU epilogue in GEMM (regression!)

Added `gemm_nn_silu_f16` kernel that applies SiLU (sigmoid × x) to the
float accumulators before converting to half during store. Eliminates
separate `silu_inplace` kernel launch and memory round-trip.

**Result:** FF1 went from 1.53ms to 1.66ms — **worse**. The `exp()` in the
store path is expensive with only 128 threads. The separate silu_inplace
kernel runs with maximum parallelism (one thread per element), which is
much faster despite the extra memory round-trip.

### 4. Stride-8 threadgroup layout (mixed, abandoned)

Reimplemented the GEMM with llama.cpp's stride-8 contiguous 8x8 block
layout for threadgroup memory instead of our row-major stride-32 layout.

**Result:** Better for [131,1024,4096] (1.13 → 1.85 TF) but worse for
[4096,4096,1024] (3.45 → 2.23 TF). The complex block-index computation
during cooperative loading outweighed the benefit of contiguous simdgroup_load.

## The diagnostic breakthrough

Wrote compute-only and load-only diagnostic kernels to isolate where time
was actually spent:

```
Shape              full ms  compute ms  load ms
[131, 4096, 1024]    0.529      0.054     0.351  (load = 66%)
[131, 1024, 4096]    0.963      0.069     0.800  (load = 83%)
[4096,4096, 1024]    9.949      1.079     6.688  (load = 67%)
```

**66-83% of GEMM time was in device→threadgroup memory loads.** Compute was
only 10%. All previous optimizations targeted the wrong thing.

## The fix: vectorized loads with bounds-check elimination

Two changes to the load loops in all GEMM kernels:

1. **`half4` vectorized loads** — replaced scalar `sa[...] = X[...]` with
   `*(threadgroup half4*)(sa + ...) = *(device const half4*)(X + ...)`,
   loading 4 halfs per instruction instead of 1.

2. **Fast-path bounds elimination** — precompute `full_m = (r0 + NR <= M)`
   and `full_k = (k + NK <= K)` before the load loop. When both are true
   (the common case for non-boundary tiles), skip all per-element bounds
   checks. The boundary tiles still use the scalar path with checks.

## Result

- **GPU encode: 125ms → 88ms (-30%)**
- **Per-block: 4.83ms → 3.36ms (-30%)**
- **RTFx: 40x → 55-62x (+50%)**
- 16/16 correctness tests pass

## GEMM throughput (isolated benchmark)

| Shape | Before (TFLOPS) | After (TFLOPS) | Improvement |
|---|---|---|---|
| FF1-expand [131,4096,1024] | 2.13 | 3.10 | +45% |
| FF1-contract [131,1024,4096] | 1.13 | 1.95 | +73% |
| Pos-proj [261,1024,1024] | 2.31 | 3.07 | +33% |
| Out-proj [131,1024,1024] | 1.24 | 1.98 | +60% |

## Remaining opportunities

- Decode path: 57ms for ~30 steps = ~2ms/step (small GEMVs, hard to optimize)
- CPU mel: 11ms could move to GPU
- NT kernel loads: the transpose load (`sb[k,n] = W[n,k]`) can't be vectorized
  easily; a NN layout for weights would help
- Higher-level: GEMM still at ~3 TFLOPS vs 10.4 peak; deeper load optimization
  (stride-8 layout with vectorized loads) or double-buffering could help further
