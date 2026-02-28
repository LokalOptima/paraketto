# parakeet

Single-file C++ speech-to-text using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model with TensorRT. ~950 lines, no Python at inference time.

```
WAV (16kHz mono) → CPU mel spectrogram (cuFFT) → TRT encoder → TRT decoder (greedy TDT) → text
```

## Performance

```
librispeech: WER=1.81%  RTFx=827x  (40 utts, 276s audio)
earnings22:  WER=16.48% RTFx=796x  (40 utts, 253s audio)
long-audio:  RTFx=1049x (92s audio, 87ms inference)
startup:     438ms warm, 1129ms cold
```

## Quickstart

Prerequisites: Linux, NVIDIA GPU, CUDA toolkit, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install Python deps (for engine building + benchmarks)
make engines                     # build TRT engines from ONNX models (~1.2GB encoder, 18MB decoder)
make parakeet                    # compile the C++ binary
./parakeet audio.wav             # transcribe a 16kHz mono WAV file
./parakeet --engine-dir DIR *.wav  # custom engine path, multiple files
```

## Benchmarks

```bash
make bench       # run both Python and C++ benchmarks
make bench-cpp   # run C++ benchmark only
```

## Project structure

```
src/parakeet.cpp         # single-file C++ inference runtime
scripts/build_engines.py # ONNX → TensorRT engine builder
engines/
  encoder.engine         # 1.2 GB, FP16 (GPU-specific, not checked in)
  decoder_joint.engine   # 18 MB, FP16
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA batch ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
