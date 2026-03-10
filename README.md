<p align="center">
  <img src="paraketto.png" width="256" alt="parakettő">
</p>

# parakettő

Speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

- Batch 1, 1300x+ real-time — fast on a single WAV
- Custom CUDA/CUTLASS kernels — only `libcudart.so`
- Optional FP8 quantization — half the weight size, +9% throughput
- Optimized 1.8 GB VRAM usage
- ~240ms warm startup (FP16), ~180ms (FP8)
- Builtin HTTP server
- Optional static build with zero runtime files

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti, batch size 1. Two FP16 GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (requires `libcublas.so`). Plus an **FP8** backend using cublasLt E4M3 quantized weights. Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in all backends.

```
                 CUTLASS (cudart only)          cuBLAS (+ libcublas)
              ────────────────────────────   ────────────────────────────
               RTFx    WER    Audio  Time     RTFx    WER    Audio  Time
librispeech   1069x   1.38%   896s  838ms    1047x   1.38%   896s  856ms
earnings22     955x  11.37%   253s  265ms     994x  11.37%   253s  255ms
long          1307x   1.62%  5578s  4.27s    1302x   1.61%  5578s  4.28s
difficult     1211x  20.99%   509s  421ms    1248x  20.99%   509s  408ms
              ────────────────────────────   ────────────────────────────
Total         1249x          7236s  5.79s    1247x          7236s  5.80s
```

FP8 backend with fused quantization (requires Blackwell GPU):

```
                 FP8 (cublasLt E4M3 + fused quantize)
              ────────────────────────────
               RTFx    WER    Audio  Time
librispeech   1228x   1.64%   896s  729ms
earnings22    1086x  11.37%   253s  233ms
long          1352x   1.82%  5578s  4.12s
difficult     1354x  16.46%   509s  376ms
              ────────────────────────────
Total         1325x          7236s  5.46s
```

### Startup time

Time from process start to first inference, measured with `tests/bench_startup.py`:

```
                startup (cold / warm)
CUTLASS:       600ms / 240ms      paraketto-fp16.bin (1.2 GB)
cuBLAS:        620ms / 240ms      paraketto-fp16.bin (1.2 GB)
FP8:           325ms / 180ms      paraketto-fp8.bin (604 MB)
```

Cold = weight files not in OS page cache. Warm = cached.

### Test machine

```
┌───────────┬────────────────────────────────────────────────────────────────┐
│ CPU       │ Intel Core i7-12700 — 2.1 GHz base / 4.9 GHz boost, 25 MB L3   │
│ RAM       │ Corsair Vengeance LPX 32 GB DDR4-3200 CL16, dual ch, 51.2 GB/s │
│ GPU       │ NVIDIA GeForce RTX 5070 Ti — 16 GB GDDR7, 896 GB/s, 2452 MHz   │
│ Storage   │ Samsung 970 EVO 1 TB NVMe — PCIe 3.0 x4, 3400/2500 MB/s r/w    │
└───────────┴────────────────────────────────────────────────────────────────┘
```

## Backends

Three CUDA backends, same driver and weight loader:

| Binary | GEMM backend | Weights | Notes |
|--------|-------------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `paraketto-fp16.bin` (1.2 GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `paraketto-fp16.bin` (1.2 GB) | |
| `paraketto.fp8` | cublasLt FP8 E4M3 | `paraketto-fp8.bin` (604 MB) | Blackwell only |

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
- `wget` (for auto-downloading weights)
- Python 3.10+ with [uv](https://docs.astral.sh/uv/) (for benchmarks only — not needed at runtime)

### Build & run

```bash
make paraketto.cuda              # CUTLASS backend (cudart only)
./paraketto.cuda audio.wav       # auto-downloads weights on first run (~1.2 GB)
```

Weights are downloaded from [HuggingFace](https://huggingface.co/localoptima/paraketto) to `~/.cache/paraketto/` on first run. Use `--weights FILE` to override with a local file.

### FP8 backend (Blackwell)

```bash
make paraketto.fp8               # build FP8 binary
./paraketto.fp8 audio.wav        # auto-downloads paraketto-fp8.bin (~604 MB)
```

## Usage

```bash
./paraketto.cuda audio.wav               # single file
./paraketto.cuda *.wav                   # multiple files
./paraketto.cuda --weights FILE audio.wav  # custom weights path
```

### Server mode

```bash
./paraketto.cuda --server                    # listen on 0.0.0.0:8080
./paraketto.cuda --server :5001              # custom port
./paraketto.cuda --server 127.0.0.1:5001     # bind to localhost
```

All backends support the same server mode.

## HTTP API

- `GET /health` — returns `{"status":"ok"}`
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`

```bash
curl localhost:8080/health
curl -F file=@audio.wav localhost:8080/transcribe
```

## Benchmarks

```bash
make bench-cuda    # WER + RTFx (CUTLASS backend)
make bench-cublas  # WER + RTFx (cuBLAS backend)
make bench-fp8     # WER + RTFx (FP8 backend)
make bench-all     # all backends

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.38% │   1069x │    100 │    896s │    838ms │
│ earnings22  │   11.37% │    955x │     40 │    253s │    265ms │
│ long        │    1.62% │   1307x │     50 │   5578s │    4.27s │
│ difficult   │   20.99% │   1211x │     50 │    509s │    421ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1249x │    240 │   7236s │    5.79s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.38% │   1047x │    100 │    896s │    856ms │
│ earnings22  │   11.37% │    994x │     40 │    253s │    255ms │
│ long        │    1.61% │   1302x │     50 │   5578s │    4.28s │
│ difficult   │   20.99% │   1248x │     50 │    509s │    408ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1247x │    240 │   7236s │    5.80s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ FP8  · paraketto_cuda.cpp + cublasLt FP8

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.64% │   1228x │    100 │    896s │    729ms │
│ earnings22  │   11.37% │   1086x │     40 │    253s │    233ms │
│ long        │    1.82% │   1352x │     50 │   5578s │    4.12s │
│ difficult   │   16.46% │   1354x │     50 │    509s │    376ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1325x │    240 │   7236s │    5.46s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘
```

## Static binary (no runtime files)

```bash
make paraketto.static      # embeds paraketto-fp16.bin, CUTLASS FP16
make paraketto.fp8.static  # embeds paraketto-fp8.bin, FP8
```

Requires only the NVIDIA driver + shared CUDA/cuBLAS libraries. No weights files at runtime.

## Project structure

```
src/paraketto_cuda.cpp    # CUDA backend main (mel, server, greedy decode)
src/conformer.cpp         # FP16 CudaModel (CUTLASS or cuBLAS via gemm.h)
src/conformer_fp8.cpp     # FP8 CudaModel (cublasLt E4M3, per-tensor scaling)
src/conformer_fp8.h       # FP8 CudaModel header (adds fp8_pool, scales, handles)
src/weights.cpp           # Weight loading (shared by all backends)
src/gemm.h                # Unified GEMM interface (backend selected at link time)
src/cutlass_gemm.cu       # CUTLASS FP16 backend
src/cublas_gemm.cu        # cuBLAS FP16 backend
src/kernels.cu            # Custom kernels: FFT, LayerNorm, SiLU, GLU, conv, LSTM, ...
src/kernels_fp8.cu        # FP8 kernels: absmax quantize, static quantize, fused FP8 output
src/mel.h                 # Custom 512-point FFT + mel filterbank
scripts/export_weights.py # NeMo → paraketto-fp16.bin converter
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA's ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA Templates for Linear Algebra Subroutines
