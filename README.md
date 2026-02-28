# parakeet

Single-file C++ speech-to-text using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model. Two backends: TensorRT and pure CUDA (cuBLAS + custom kernels).

```
WAV (16kHz mono) → mel spectrogram (cuFFT) → conformer encoder → TDT greedy decoder → text
```

## Performance

```
             RTFx     WER (libri)   startup
TRT:         638x     1.81%         392ms
CUDA:        600x     1.81%         238ms
```

## parakeet.cuda (recommended)

Pure CUDA backend — no TensorRT or cuDNN dependency. Only requires CUDA toolkit (cudart, cuBLAS, cuFFT).

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
- Python 3.10+, [uv](https://docs.astral.sh/uv/) (for weight export + benchmarks only)

### Build & run

```bash
uv sync                          # install Python deps (for weight export)
make weights                     # export model weights to weights.bin (~1.2GB)
make parakeet.cuda               # compile the CUDA binary
./parakeet.cuda audio.wav        # transcribe a 16kHz mono WAV file
./parakeet.cuda *.wav            # multiple files
./parakeet.cuda --weights FILE audio.wav  # custom weights path
```

### Server mode

```bash
./parakeet.cuda --server              # listen on 0.0.0.0:8080
./parakeet.cuda --server :5001        # listen on 0.0.0.0:5001
./parakeet.cuda --server 127.0.0.1:5001  # bind to localhost only
```

### Benchmarks

```bash
make bench-cuda  # WER + RTFx on librispeech and earnings22
```

## parakeet (TensorRT backend)

Reference TensorRT backend. Requires TensorRT runtime libraries.

```bash
uv sync                          # install Python deps (includes TensorRT)
make engines                     # build TRT engines from ONNX models
make parakeet                    # compile the TRT binary
./parakeet audio.wav             # transcribe
```

```bash
make bench-cpp   # benchmark TRT backend
```

## HTTP API

Both backends support the same server mode and HTTP API:

- `GET /health` — returns `{"status":"ok"}`
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`

```bash
curl localhost:8080/health
curl -F file=@audio.wav localhost:8080/transcribe
```

## Project structure

```
src/parakeet_cuda.cpp    # CUDA backend main (mel + server + greedy decode)
src/conformer.cpp        # conformer encoder + decoder (cuBLAS + custom kernels)
src/kernels.cu           # custom CUDA kernels (LayerNorm, SiLU, GLU, conv, LSTM, etc.)
src/parakeet.cpp         # TensorRT backend (reference)
scripts/export_weights.py  # NeMo → weights.bin converter
scripts/build_engines.py   # ONNX → TRT engine builder
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA batch ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
