<p align="center">
  <img src="paraketto.png" width="256" alt="parakettő">
</p>

# parakettő

Speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

- **V2** (English) and **V3** (25 EU languages, auto-detect) — `--model v3`
- Batch 1, 1200x–1400x real-time — fast on a single WAV
- Custom CUDA/CUTLASS kernels — only `libcudart.so`
- Long audio support — files >120s auto-split at silence boundaries
- Low VRAM: approximately 1.8 GB
- ~240ms warm startup
- Builtin HTTP server with web UI (microphone + file upload)

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti, batch size 1. Two FP16 GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (requires `libcublas.so`). Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in all backends.

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

### Startup time

Time from process start to first inference, measured with `tests/bench_startup.py`:

```
                startup (cold / warm)
CUTLASS:       600ms / 240ms      paraketto-fp16.bin (1.2 GB)
cuBLAS:        620ms / 240ms      paraketto-fp16.bin (1.2 GB)
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

Two FP16 CUDA backends, same driver and weight loader. Both support V2 (English) and V3 (multilingual) via `--model v2|v3`:

| Binary | GEMM backend | Weights | Notes |
|--------|-------------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `paraketto-fp16.bin` (1.2 GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `paraketto-fp16.bin` (1.2 GB) | |

V3 weights: `paraketto-v3-fp16.bin` (1.2 GB), auto-downloaded on the first `--model v3` run.

FP8 inference, quantization, calibration, and kernel experiments are preserved
on the separate `fp8-ultra` branch. The main branch uses FP16.

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
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

The FP16 GEMMs come from the [CUTLASS](https://github.com/NVIDIA/cutlass) submodule at `third_party/cutlass` (header-only — nothing to build). The GPU architecture is auto-detected via `nvidia-smi`; override with `make ARCH=sm_XX` if cross-compiling. The optional LLM corrector additionally needs the `vendor/llama.cpp` submodule.

Weights are downloaded from [HuggingFace](https://huggingface.co/localoptima/paraketto) to `~/.cache/paraketto/` on first run. Use `--weights FILE` to override with a local file.

### Multilingual (V3)

```bash
./paraketto.cuda --model v3 audio.wav   # 25 EU languages, auto-detect
./paraketto.cublas --model v3 audio.wav # same model, cuBLAS backend
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
single warmed pass; this is not a statistical speed comparison. Multilingual
WER uses the HF Open ASR Leaderboard normalizer with `num2words` expansion.

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

Recordings and uploads stay in a stack, newest first, with playback and a copy
button on each transcript. History lasts until you refresh or close the page.

### Linux desktop dictation

The optional native app connects to a persistent local Ultra FP16 service. Its window follows
the desktop theme and text scale. History fills the window, grouped by day, with
service controls and quiet stats in the footer. Transcripts initially show at
most four visual lines; a small chevron expands or collapses longer entries.
Playback, delete, and copy stay beside each entry, and Copy always uses the full
transcript. The shortcut is shown as **Ctrl+Alt+Space**. The server row uses bold yellow **Sleeping**, green **Ready**,
and **Loading…** while the model loads; the service does not report loading progress.
Status, VRAM, and action columns keep their positions across state changes.
The padded, monospaced VRAM display shows
**process usage / GPU capacity** in GiB, including zero usage when unloaded;
hover over it for device-wide usage while the server is running.
The stats pane shows total words in saved transcriptions and whole-number WPM
(total words divided by their total recording time in minutes).

The service starts with the model **Sleeping**, with no model GPU allocation.
Recording starts immediately and loads the model in the background. By default,
the service unloads it after **5 minutes without activity**. Preferences provides
the idle timeout and **Keep loaded between recordings**. **Load**, **Unload**,
and **Retry** are available in the server row; unloading is disabled
during recording or transcription by any client. Unloading stops the private
GPU worker, releasing its CUDA context as well as model allocations. The API
stays available while the model sleeps or fails to load. It prefers
`http://127.0.0.1:8080` and automatically chooses a free port if 8080 is occupied.
The desktop discovers the address, including after a service restart, and never
starts or stops the service itself. Preferences shows the current address.

A failed load shows a simple red **Couldn’t load model** message in the window
and active dictation overlay. Recording stops, any retained audio stays in
history, and details are written to the server log. The error overlay stays
until dismissed or another dictation is started.

Build dependencies on Ubuntu/Debian (these install system packages):

```bash
sudo apt install libgtk-3-dev libibus-1.0-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev libjson-glib-dev libxdo-dev
make desktop
make install-service
./paraketto-desktop --install-desktop
./paraketto-desktop
```

`make install-service` installs and starts a **user systemd service**, configured
to start with your login session. It needs no sudo. Its executable remains in
this checkout; rerun the installer if you move the checkout. For a foreground
service instead, run `./paraketto-service` in another terminal.

`--install-desktop` adds a launcher to your user applications directory and a
GNOME **Ctrl+Alt+Space** shortcut, preserving existing custom shortcuts. It does
not enable autostart. Change the binding in GNOME Settings → Keyboard → Custom
Shortcuts, or use Preferences to enable the default binding again.

Focus a text field and press the shortcut to record. A floating audio-level
indicator shows elapsed time and the destination's desktop application name
(for example, **Ghostty**) without taking keyboard focus. Press the same
shortcut to stop; the transcript is inserted when recognition finishes. The
indicator's round × button beside the waveform cancels recording or pending insertion. The main window's
Record button records directly into history.

IBus support is checked for the focused input on each dictation. A temporary
engine supplies native Unicode text delivery and then restores the previous
engine. Applications without a confirmed IBus input use X11 clipboard/paste;
that fallback leaves the transcript on the clipboard. A changed destination
retains the result in history for manual copying. Native delivery tracks field
focus; the X11 fallback can only track the application window. Initial desktop
support targets GNOME on X11; Wayland integration is not yet validated.

History and WAV recordings persist in `~/.local/share/paraketto/history/`
(`$XDG_DATA_HOME/paraketto/history/` when set). Use a recording's trash button to delete both.
The service owns memory preferences in `~/.local/share/paraketto/preferences.ini`;
existing preferences are preserved. Desktop preferences update them through the API.
The server log is `~/.local/share/paraketto/server.log`. Closing the window
keeps dictation available; choose **Quit Paraketto** from the menu, or run
`./paraketto-desktop --quit`, to stop the app. The service and other clients keep
working. Stop the service with `systemctl --user stop paraketto.service`;
start it again with `systemctl --user start paraketto.service`.

Weights default to `paraketto-ultra-fp16.bin` beside the executable, then
`~/.cache/paraketto/paraketto-ultra-fp16.bin`. The **service** accepts `--weights FILE`,
`--worker-binary FILE`, `--data-dir DIRECTORY`, `--port PORT`, and `--load-timeout SECONDS`
(default 120). An explicit `--port PORT` pins the address and fails if that port
is occupied; `--port 0` always requests a free port. The **desktop** accepts
`--service-url URL` to override discovery and `--data-dir DIRECTORY` for its
history and local service discovery. When using a custom service data directory,
give both processes the same `--data-dir`, or supply the desktop's `--service-url`.
The menu can import existing mono 16/24 kHz WAV recordings.

The app reuses GTK, GStreamer, IBus, libxdo, and the existing cpp-httplib server.
[Speech Note](https://github.com/mkiol/dsnote), with its maintained stable 4.x
releases, was the main desktop-integration reference; its
[active-window implementation](https://github.com/mkiol/dsnote/blob/main/src/fake_keyboard.cpp)
uses native X11 input and clipboard fallbacks. We also reviewed
[Nerd Dictation](https://github.com/ideasman42/nerd-dictation)'s recording/input
separation. The desktop code lives in `src/desktop/`; no Python is needed at
runtime. The service lives in `src/service/` and only needs GIO and JSON-GLib,
without GTK or CUDA linkage. CMake users can opt in with
`-DPARAKETTO_BUILD_DESKTOP=ON` (includes the service) or `-DPARAKETTO_BUILD_SERVICE=ON`.

`make test-desktop` checks recording, persistent history, native IBus delivery,
focus changes, and private-worker cleanup in a private X11/D-Bus session. It also
requires the `xvfb` package. The test session does not change the active desktop's
input engine or keyboard shortcuts.

`make test-service` checks API availability while unloaded, multiple clients,
bounded queues, idle/manual unloading, recording reservations, persisted settings,
failed loads, worker crashes, startup deadlines, port collisions, and shutdown. `make test-desktop-lifecycle`
checks the real desktop's API integration, recording reservations, quitting the
UI during another client's request, discovery after a port change, stale-address
rejection, and recovery after a service restart. Both
use `uv` and a Python HTTP fixture; they do not load a GPU model.

### Local service API

The persistent service binds **only to loopback**. This milestone does not configure
remote access or authentication. Browser requests from other origins are rejected,
and process shutdown is controlled by systemd/signals rather than an HTTP endpoint.
The existing standalone `paraketto.cuda --server` mode remains available separately.

The service atomically publishes its port and PID in
`~/.local/share/paraketto/service.json` (or its `--data-dir`), readable only by
the current user, and removes it on clean shutdown. The desktop checks service
identity, API version, and PID before sending audio or control requests, so a
stale discovery file cannot silently select another service instance. An explicit
`--service-url` checks identity/version without depending on the discovery file.

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Service identity and API version, independent of model readiness |
| `GET /status` | Model state, GPU memory, active/queued requests, recording reservations, idle countdown, and settings |
| `GET /info` | Last known model/backend metadata; available without loading the model |
| `POST /transcribe` | Multipart `file` containing a WAV; loads automatically and returns the existing transcription JSON |
| `POST /model/load` | Preload asynchronously; returns `202` |
| `POST /model/unload` | Unload or cancel an idle load; `409` if a client is recording or work is queued/active |
| `PATCH /settings` | JSON `idle_seconds` (1–86400) and/or boolean `keep_loaded`, persisted by the service |
| `POST /model/release` | Release the caller's recording reservation |

States are `sleeping`, `loading`, `ready`, `busy`, `unloading`, `error`, and `stopping`.
An offline service is a separate client-side condition. Status includes
`service: "paraketto"` and integer `api_version: 1`. Errors use JSON `error` messages.
Requests share one GPU worker and are serialized, with a maximum of four accepted
transcriptions (including the active request); excess requests receive `429`.
The upload limit is 64 MiB. Model-load failure returns `503` without stopping the API.

Recording clients may pass `{"lease_id":"<UUID>"}` to `POST /model/load` and renew
it while recording. Reservations expire after 10 seconds unless renewed; the
desktop renews every 2 seconds and releases after cancellation/delivery. Send
the same JSON to `POST /model/release` when finished. This prevents idle eviction
while recording and permits recovery if a client crashes. Transcription requests
protect the model automatically and do not require a reservation.

Use the address shown in Preferences; these examples assume port 8080:

```bash
curl http://127.0.0.1:8080/status
curl -F file=@recording.wav http://127.0.0.1:8080/transcribe
curl -X POST http://127.0.0.1:8080/model/unload
```

### LLM text correction (optional)

Build with `WITH_CORRECTOR=1` to enable post-transcription text correction via an embedded LLM (OLMoE-1B-7B). Removes filler words, fixes capitalization/punctuation.

```bash
make paraketto.cuda WITH_CORRECTOR=1    # build with llama.cpp integration
./paraketto.cuda --correct audio.wav    # enable correction in CLI mode
./paraketto.cuda --server               # correction auto-enabled in server mode
```

Requires the llama.cpp git submodule (`git submodule update --init`).

### Embedding as a library (CMake)

The Makefile is the canonical build for the standalone binaries. A separate CMake build produces a static library (`paraketto_lib`) for embedding the engine in other projects — see the header comment in `CMakeLists.txt` for options (`PARAKETTO_USE_CUBLAS`, `PARAKETTO_BUILD_LIB_ONLY`). The public API is `paraketto::Pipeline` in `src/paraketto.h`. The CMake build does not support the LLM corrector.

## HTTP API

- `GET /` — web UI (microphone recording + file upload; shows the running model/backend)
- `GET /health` — returns `{"status":"ok"}`
- `GET /info` — model version, backend (CUTLASS FP16/cuBLAS FP16), weights path, device, startup timing
- `GET /status` — readiness, server PID, live server-process and device-wide VRAM (NVML; unavailable values are omitted or `null`)
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`
- `POST /shutdown` — gracefully stops the server

```bash
curl localhost:8080/health
curl localhost:8080/info
curl localhost:8080/status
curl -F file=@audio.wav localhost:8080/transcribe
curl -X POST localhost:8080/shutdown
```

For managed clients, `--server 127.0.0.1:0` binds an available loopback port and
prints a JSON announcement to stdout: `{"event":"ready","port":…,"pid":…}`.
Clients should then check `/status`. Request logs remain on stderr.

## Benchmarks

```bash
make bench-cuda    # WER + RTFx (CUTLASS backend)
make bench-cublas  # WER + RTFx (cuBLAS backend)
make bench-all     # both FP16 backends
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
```

Correctness checks:

```bash
uv run python -m unittest discover -s tests -p test_ultra_conversion.py
make test-cublas-fallback
```

## Project structure

```
src/paraketto_cuda.cpp    # CLI, startup, and HTTP server entry point
src/paraketto.h           # Public pipeline API, chunking, and greedy decode
src/model_defs.h          # Shared model constants, config, Weights struct
src/conformer.h           # FP16 CudaModel definition
src/conformer.cpp         # Buffer pool, FP16 encoder, LSTM and joint decoder
src/weights.cpp           # Weight loading (shared by all backends)
src/gemm.h                # Unified GEMM interface (backend selected at link time)
src/cutlass_gemm.cu       # CUTLASS FP16 backend
src/cublas_gemm.cu        # cuBLAS FP16 backend
src/kernels.cu            # Custom kernels: FFT, LayerNorm, SiLU, GLU, conv, LSTM, ...
src/mel.h                 # Custom 512-point FFT + mel filterbank
src/wav.h                 # WAV file reader (16kHz/24kHz mono, int16/float32)
src/server.h              # HTTP server + web UI (uses cpp-httplib)
src/corrector.cpp         # Optional LLM text correction (llama.cpp integration)
scripts/weight_format.py  # Shared FP16 tensor order and alignment
scripts/export_weights.py # ONNX → PRKT FP16 converter (V2 / V3)
scripts/export_ultra_weights.py # Safetensors → PRKT FP16 converter (Ultra)
```

## Acknowledgments

- **[Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** by NVIDIA — the original ASR model ([V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) English, [V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) multilingual). This project is a from-scratch C++/CUDA reimplementation of the inference pipeline. Model weights are used under CC-BY-4.0, Copyright NVIDIA Corporation.
- **[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)** by Moondream — a post-trained V3 checkpoint, also CC-BY-4.0. The optional FP16 converter preserves our V3 inference layout.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA Templates for Linear Algebra Subroutines, used as a git submodule for FP16 GEMMs (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — optional LLM text correction via git submodule (MIT License, Copyright 2023-2026 The ggml authors)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server (MIT License)

## References

- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [V3 paper](https://arxiv.org/abs/2509.14128) — Canary-1B-v2 & Parakeet-TDT-0.6B-v3
