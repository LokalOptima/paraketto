<p align="center">
  <img src="paraketto.png" width="256" alt="parakettő">
</p>

# parakettő

Speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

- **V2** (English) and **V3** (25 EU languages, auto-detect) — `--model v3`
- Batch 1, 1200x–1400x real-time — fast on a single WAV
- Custom CUDA/CUTLASS kernels — only `libcudart.so`
- Optional FP8 quantization (CUTLASS E4M3) — half the weight size, ~35% less VRAM
- Long audio support — files >120s auto-split at silence boundaries
- Low VRAM: 1.8 GB (FP16), 1.2 GB (FP8)
- ~240ms warm startup (FP16), ~180ms (FP8)
- Builtin HTTP server with web UI (microphone + file upload)

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti, batch size 1. Two FP16 GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (requires `libcublas.so`). Plus an **FP8** backend using CUTLASS E4M3 quantized weights. Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in all backends.

```
                 CUTLASS (cudart only)          cuBLAS (+ libcublas)
              ────────────────────────────   ────────────────────────────
               RTFx    WER    Audio  Time     RTFx    WER    Audio  Time
librispeech   1109x   1.38%   896s  808ms    1048x   1.38%   896s  854ms
earnings22     957x  11.37%   253s  264ms     973x  11.37%   253s  260ms
long          1355x   1.64%  5578s  4.12s    1356x   1.63%  5578s  4.11s
difficult     1189x  20.99%   509s  428ms    1223x  21.07%   509s  416ms
              ────────────────────────────   ────────────────────────────
Total         1288x          7236s  5.62s    1282x          7236s  5.64s
```

FP8 backend with fused quantization (requires Blackwell GPU):

```
                 FP8 (CUTLASS E4M3 + fused quantize)
              ────────────────────────────
               RTFx    WER    Audio  Time
librispeech   1043x   1.42%   896s  859ms
earnings22     979x  11.82%   253s  259ms
long          1387x   1.81%  5578s  4.02s
difficult     1210x  16.46%   509s  421ms
              ────────────────────────────
Total         1301x          7236s  5.56s
```

V3 multilingual (FP8, FLEURS test clips, 50 per language):

```
                 V3 FP8 multilingual (CUTLASS E4M3)
              ────────────────────────────
               RTFx    WER    Audio  Time
german        1256x   7.90%   695s  553ms
italian       1490x   3.90%   732s  491ms
french        1158x   4.84%   498s  431ms
              ────────────────────────────
Total         1305x          1925s  1.47s
```

WER uses the [HF Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) multilingual normalizer (lowercase + strip diacritics + remove punctuation) with `num2words` number expansion. The FP8 activation scales are calibrated with 1.5× headroom over a broad audio set (`scripts/calibrate_v3_scales.py`); these numbers match the FP16 backend within 0.3 point and produce no scale-induced empty transcriptions — 0 across the full Common Voice test set (38,918 clips de/it/fr).

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

Three CUDA backends, same driver and weight loader. All support both V2 (English) and V3 (multilingual) via `--model v2|v3`:

| Binary | GEMM backend | Weights | Notes |
|--------|-------------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `paraketto-fp16.bin` (1.2 GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `paraketto-fp16.bin` (1.2 GB) | |
| `paraketto.fp8` | CUTLASS FP8 E4M3 | `paraketto-fp8.bin` (604 MB) | Blackwell only |

V3 weights: `paraketto-v3-fp16.bin` (1.2 GB) / `paraketto-v3-fp8.bin` (627 MB). Auto-downloaded on first `--model v3` run.

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer; FP8 backend needs Blackwell), CUDA toolkit 12+
- `curl` (for auto-downloading weights at runtime)
- `wget` (for Makefile benchmark data downloads)
- Python 3.10+ with [uv](https://docs.astral.sh/uv/) (for benchmarks only — not needed at runtime)

### Build & run

```bash
git clone --recurse-submodules https://github.com/LokalOptima/paraketto
# or, in an existing clone: git submodule update --init third_party/cutlass
make paraketto.cuda              # CUTLASS backend (cudart only)
./paraketto.cuda audio.wav       # auto-downloads weights on first run (~1.2 GB)
```

The FP16/FP8 GEMMs come from the [CUTLASS](https://github.com/NVIDIA/cutlass) submodule at `third_party/cutlass` (header-only — nothing to build). The GPU architecture is auto-detected via `nvidia-smi`; override with `make ARCH=sm_XX` if cross-compiling. The optional LLM corrector additionally needs the `vendor/llama.cpp` submodule.

Weights are downloaded from [HuggingFace](https://huggingface.co/localoptima/paraketto) to `~/.cache/paraketto/` on first run. Use `--weights FILE` to override with a local file.

### FP8 backend (Blackwell)

```bash
make paraketto.fp8               # build FP8 binary
./paraketto.fp8 audio.wav        # auto-downloads paraketto-fp8.bin (~604 MB)
```

### Multilingual (V3)

```bash
./paraketto.fp8 --model v3 audio.wav    # 25 EU languages, auto-detect
./paraketto.cuda --model v3 audio.wav   # works with any backend
```

Supports: bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk.

### Parakeet Ultra

[Moondream Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) is a
fine-tuned V3 checkpoint with the same architecture and vocabulary. Convert its
Safetensors checkpoint to our V3 binary format, then use either FP16 backend:

```bash
uvx hf download moondream/parakeet-ultra \
  --revision 73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b \
  --local-dir scratch/parakeet-ultra
uv run python scripts/export_ultra_weights.py scratch/parakeet-ultra
./paraketto.cuda --model v3 --weights paraketto-ultra-fp16.bin audio.wav
```

The converter checks the configuration, every tensor shape, and the vocabulary;
transposes linear weights; folds convolution batch normalization; and reorders
the LSTM gates. It needs no PyTorch or NeMo. Output is approximately 1.25 GB.
The binary identifies this as V3; the weights path distinguishes Ultra.

Ultra uses our existing decoder and silence-based long-audio chunking. Its
auxiliary VAD head is not imported, so this does not reproduce Photon's exact
inference pipeline. FP16 needs no calibration.

Compare against the original V3 weights on the same local benchmark clips:

```bash
uv run python tests/bench_weights.py \
  --weights paraketto-ultra-fp16.bin \
  --baseline ~/.cache/paraketto/paraketto-v3-fp16.bin \
  --output scratch/ultra-benchmark.json
```

The default comparison uses the existing English and de/it/fr benchmark sets.
`--datasets` selects other local manifests, including `fleurs_german`,
`fleurs_italian`, `fleurs_french`, and `cv_{de,it,fr}_heldout` when available.
The JSON includes transcripts, WER, empty-output counts, and weight checksums.

FP16 comparison on RTX 5070 Ti, 2026-10-27 (WER %, lower is better):

| Local dataset | Clips | V3 | Ultra |
|---|---:|---:|---:|
| LibriSpeech | 100 | 1.94 | 1.64 |
| Earnings22 | 40 | 12.41 | 10.49 |
| Long audio | 50 | 1.92 | 1.78 |
| Difficult | 50 | 20.74 | 18.56 |
| FLEURS German | 347 | 6.99 | 6.43 |
| FLEURS Italian | 346 | 4.97 | 4.19 |
| FLEURS French | 332 | 6.95 | 6.63 |
| Common Voice German, held out | 25 | 0.46 | 0.46 |
| Common Voice Italian, held out | 25 | 2.90 | 2.49 |
| Common Voice French, held out | 25 | 3.77 | 3.77 |

Both models produced zero empty transcripts across 1,340 clips (5.85 hours).
Aggregate throughput was 1,152× real time for V3 and 1,150× for Ultra in a
single warmed pass; this is not a statistical speed comparison. These local
FLEURS sets are larger than the 50-clip sets in the V3 FP8 table above.

#### Ultra FP8 (Blackwell)

Convert the FP16 file with the existing E4M3 weight quantizer, then calibrate
activation ranges using Ultra's own weights. Calibration is an inference pass;
there is no retraining.

```bash
make paraketto.fp8
uv run python scripts/generate_fp8_weights.py \
  paraketto-ultra-fp16.bin paraketto-ultra-fp8.bin
uv run python scripts/calibrate_fp8.py \
  --weights paraketto-ultra-fp8.bin \
  --output paraketto-ultra-fp8-calibrated.bin \
  --report scratch/ultra-fp8-calibration.json
./paraketto.fp8 --model v3 --weights paraketto-ultra-fp8-calibrated.bin audio.wav
```

The calibrator requires the local corpora listed in `scripts/calibrate_fp8.py`:
162 clips from Common Voice (en/de/it/fr), FLEURS (de/it/fr), LibriSpeech,
Earnings22, and difficult speech. It dynamically rescales every FP8 GEMM input
while measuring all 218 activation sites, takes their maxima across clips and
chunks, and applies 1.5× headroom. The output bundles the scales with the weights
in PRKTFP8 format 3 (657.3 MB). Original format-2 V2/V3 files continue to use
their baked scales. Use the **calibrated** file for Ultra inference; the
intermediate file still selects the original V3 scales.

To compare FP8 against FP16, excluding calibration recordings by decoded-audio
hash even if they appear under different filenames:

```bash
uv run python tests/bench_weights.py \
  --baseline-binary paraketto.cuda --baseline paraketto-ultra-fp16.bin \
  --binary paraketto.fp8 --weights paraketto-ultra-fp8-calibrated.bin \
  --exclude-calibration scratch/ultra-fp8-calibration.json \
  --datasets librispeech earnings22 long difficult \
    fleurs_german fleurs_italian fleurs_french \
    cv_de_heldout cv_it_heldout cv_fr_heldout \
  --output scratch/ultra-fp8-benchmark.json
```

On the same RTX 5070 Ti, after excluding 62 calibration recordings:

| Local dataset | Clips | Ultra FP16 WER | Ultra FP8 WER |
|---|---:|---:|---:|
| LibriSpeech | 88 | 1.76% | 1.91% |
| Earnings22 | 32 | 10.69% | 10.51% |
| Long audio | 50 | 1.78% | 1.81% |
| Difficult | 38 | 18.26% | 18.05% |
| FLEURS German | 337 | 6.39% | 6.30% |
| FLEURS Italian | 336 | 4.26% | 4.34% |
| FLEURS French | 322 | 6.69% | 6.77% |
| Common Voice German, held out | 25 | 0.46% | 0.46% |
| Common Voice Italian, held out | 25 | 2.49% | 2.49% |
| Common Voice French, held out | 25 | 3.77% | 3.77% |

Both produced zero empty transcripts on 1,278 evaluation clips (5.67 hours).
Aggregate throughput was 1,163× for FP16 and 1,169× for FP8 in single warmed
passes; no speed advantage is established by this measurement. GPU memory in
the server's post-warmup snapshot was 1,889 MiB vs 1,205 MiB, and the weight file
is 47.6% smaller. Validation covers English, German, Italian, and French.

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

All backends support the same server mode. Open `http://localhost:8080` in a browser for a web UI with microphone recording and file upload.

### LLM text correction (optional)

Build with `WITH_CORRECTOR=1` to enable post-transcription text correction via an embedded LLM (OLMoE-1B-7B). Removes filler words, fixes capitalization/punctuation.

```bash
make paraketto.fp8 WITH_CORRECTOR=1    # build with llama.cpp integration
./paraketto.fp8 --correct audio.wav    # enable correction in CLI mode
./paraketto.fp8 --server               # correction auto-enabled in server mode
```

Requires the llama.cpp git submodule (`git submodule update --init`).

### Embedding as a library (CMake)

The Makefile is the canonical build for the standalone binaries. A separate CMake build produces a static library (`paraketto_lib`) for embedding the engine in other projects — see the header comment in `CMakeLists.txt` for options (`PARAKETTO_FP8`, `PARAKETTO_USE_CUBLAS`, `PARAKETTO_BUILD_LIB_ONLY`). The public API is `paraketto::Pipeline` in `src/paraketto.h`. The CMake build does not support the LLM corrector.

## HTTP API

- `GET /` — web UI (microphone recording + file upload; shows the running model/backend)
- `GET /health` — returns `{"status":"ok"}`
- `GET /info` — model version, backend (FP8/FP16/cuBLAS), weights path, device, startup timing
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`
- `POST /shutdown` — gracefully stops the server

```bash
curl localhost:8080/health
curl localhost:8080/info
curl -F file=@audio.wav localhost:8080/transcribe
curl -X POST localhost:8080/shutdown
```

## Benchmarks

```bash
make bench-cuda    # WER + RTFx (CUTLASS backend)
make bench-cublas  # WER + RTFx (cuBLAS backend)
make bench-fp8     # WER + RTFx (FP8 backend)
make bench-all     # all backends
make bench-v3      # WER + RTFx (V3 multilingual: de/it/fr)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.38% │   1109x │    100 │    896s │    808ms │
│ earnings22  │   11.37% │    957x │     40 │    253s │    264ms │
│ long        │    1.64% │   1355x │     50 │   5578s │    4.12s │
│ difficult   │   20.99% │   1189x │     50 │    509s │    428ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1288x │    240 │   7236s │    5.62s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.38% │   1048x │    100 │    896s │    854ms │
│ earnings22  │   11.37% │    973x │     40 │    253s │    260ms │
│ long        │    1.63% │   1356x │     50 │   5578s │    4.11s │
│ difficult   │   21.07% │   1223x │     50 │    509s │    416ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1282x │    240 │   7236s │    5.64s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ FP8  · paraketto_cuda.cpp + CUTLASS FP8

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.42% │   1043x │    100 │    896s │    859ms │
│ earnings22  │   11.82% │    979x │     40 │    253s │    259ms │
│ long        │    1.81% │   1387x │     50 │   5578s │    4.02s │
│ difficult   │   16.46% │   1210x │     50 │    509s │    421ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1301x │    240 │   7236s │    5.56s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘
```

## Project structure

```
src/paraketto_cuda.cpp    # CUDA backend main (mel, server, greedy decode)
src/model_defs.h          # Shared model constants, config, Weights struct
src/conformer_base.h      # CudaModelBase: buffers + TDT decoder shared by FP16/FP8
src/conformer_base.cpp    # Shared buffer pool carving + decoder (LSTM, joint network)
src/conformer.h           # FP16 CudaModel definition
src/conformer.cpp         # FP16 encoder (CUTLASS or cuBLAS via gemm.h)
src/conformer_fp8.h       # FP8 CudaModel (adds fp8_pool, quantized weights, scales)
src/conformer_fp8.cpp     # FP8 CudaModel (CUTLASS E4M3, per-tensor scaling)
src/weights.cpp           # Weight loading (shared by all backends)
src/gemm.h                # Unified GEMM interface (backend selected at link time)
src/cutlass_gemm.cu       # CUTLASS FP16 backend
src/cutlass_gemm_fp8.cu   # CUTLASS FP8 E4M3 backend
src/cublas_gemm.cu        # cuBLAS FP16 backend
src/kernels.cu            # Custom kernels: FFT, LayerNorm, SiLU, GLU, conv, LSTM, ...
src/kernels_fp8.cu        # FP8 kernels: absmax quantize, static quantize, fused FP8 output
src/mel.h                 # Custom 512-point FFT + mel filterbank
src/wav.h                 # WAV file reader (16kHz/24kHz mono, int16/float32)
src/server.h              # HTTP server + web UI (uses cpp-httplib)
src/corrector.cpp         # Optional LLM text correction (llama.cpp integration)
scripts/export_weights.py # NeMo → paraketto-fp16.bin converter
```

## Acknowledgments

- **[Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** by NVIDIA — the original ASR model ([V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) English, [V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) multilingual). This project is a from-scratch C++/CUDA reimplementation of the inference pipeline. Model weights are used under CC-BY-4.0, Copyright NVIDIA Corporation.
- **[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)** by Moondream — a post-trained V3 checkpoint, also CC-BY-4.0. The optional FP16 converter preserves our V3 inference layout.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA Templates for Linear Algebra Subroutines, used as a git submodule for FP16 and FP8 GEMMs (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — optional LLM text correction via git submodule (MIT License, Copyright 2023-2026 The ggml authors)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server (MIT License)

## References

- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [V3 paper](https://arxiv.org/abs/2509.14128) — Canary-1B-v2 & Parakeet-TDT-0.6B-v3
