<p align="center">
  <img src="paraketto.png" width="256" alt="parakettő">
</p>

# parakettő

Speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

- **V2**, **V3** and **Ultra** — select architecture and weights with `--model v2|v3|ultra`
- Batch 1, up to approximately 1400x real-time in the recorded V2 benchmarks below
- Custom CUDA/CUTLASS kernels — only `libcudart.so`
- Long audio support — files >120s auto-split at silence boundaries
- Low VRAM: approximately 1.8 GB
- ~240ms warm startup
- Builtin HTTP server with web UI (microphone + file upload)

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

Recorded V2 results on RTX 5070 Ti, batch size 1. Two FP16 GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (requires `libcublas.so`). Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in all backends. The separate Ultra comparison below uses different weights; these tables are not desktop Stop-to-clipboard latency measurements.

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

Two FP16 CUDA backends, same driver and weight loader. Both support all three model presets:

| Binary | GEMM backend | Weights | Notes |
|--------|-------------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `paraketto-fp16.bin` (1.2 GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `paraketto-fp16.bin` (1.2 GB) | |

| Model flag | Architecture | Default weights |
|---|---|---|
| `--model v2` | V2, English | `paraketto-fp16.bin` |
| `--model v3` | V3, multilingual | `paraketto-v3-fp16.bin` |
| `--model ultra` | V3, multilingual; Moondream checkpoint | `paraketto-ultra-fp16.bin` |

The standalone CLI defaults to V2; the desktop service defaults to Ultra.
The selected filename is looked up beside the executable, then in
`$XDG_CACHE_HOME/paraketto/` (normally `~/.cache/paraketto/`). `--weights FILE`
overrides this lookup. An explicit `--model` checks the file's architecture
before CUDA starts. With only `--weights`, the architecture is detected from
the file and the checkpoint is reported as custom.

The standalone CLI downloads missing V2/V3 weights to the cache. Ultra needs the
one-time [conversion below](#parakeet-ultra); its converted binary is not hosted
in our download repository. The desktop service requires weights to be prepared
before loading any preset.

FP8 inference, quantization, calibration, and kernel experiments are preserved
on the separate `fp8-ultra` branch. The main branch uses FP16.

Word timestamps, forced alignment and clickable word playback are isolated on
`codex/word-timestamps`. That branch contains `docs/word-playback.md` and its
setup instructions; the main branch does not load or require Qwen alignment weights.

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
- A C++17 compiler and Make; CMake and `pkg-config` for desktop/service builds
- `curl` (for auto-downloading weights at runtime)
- `wget` (for Makefile benchmark data downloads)
- Python 3.10+ with [uv](https://docs.astral.sh/uv/) for benchmarks, weight conversion and model preparation; not needed for inference

### Build & run

```bash
git clone --recurse-submodules https://github.com/LokalOptima/paraketto
cd paraketto
# or, in an existing clone: git submodule update --init third_party/cutlass
make paraketto.cuda              # CUTLASS backend (cudart only)
./paraketto.cuda audio.wav       # auto-downloads weights on first run (~1.2 GB)
```

The FP16 GEMMs come from the [CUTLASS](https://github.com/NVIDIA/cutlass) submodule at `third_party/cutlass` (header-only — nothing to build). The GPU architecture is auto-detected via `nvidia-smi`; override with `make ARCH=sm_XX` if cross-compiling. The desktop service uses CPU ggml from the pinned `vendor/llama.cpp` submodule for speech detection.

Missing V2/V3 weights are downloaded from [HuggingFace](https://huggingface.co/localoptima/paraketto)
to the cache on first run. Use `--model` to choose a preset and `--weights FILE`
to override its file.

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
./paraketto.cuda --model ultra audio.wav
```

The converter checks the configuration, every tensor shape, and the vocabulary;
transposes linear weights; folds convolution batch normalization; and reorders
the LSTM gates. It needs no PyTorch or NeMo. Output is approximately 1.25 GB.
The preset is reported as Ultra and the architecture as V3. The binary header
verifies architecture compatibility; it cannot distinguish Ultra from another
V3 checkpoint. An explicit weights override remains the caller's choice.

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

The native GTK app connects to a persistent local service for long recordings.
The [implementation plan](docs/long-form-transcription-plan.md) records the design
and requirements. Capture belongs to the service: audio is saved continuously,
transcription runs during recording, and your dictionary applies explicit spelling
replacements to each finished speech group. Closing or crashing GTK does not discard a recording.

**Transcription** is the main view, alongside **History** and **Dictionary** tabs.
Dictionary has **Add entry**, search across spellings and aliases, and **Change**
and **Remove** on each entry. Replacements are highlighted in the transcript,
with the original wording on hover.
Select a word or phrase to open **Desired spelling** above the selection. Saving
corrects that section and remembers the alias for future transcription; the
original text remains available. Leading and trailing whitespace in the selection
is ignored; spaces inside a phrase are preserved.
Click the pill beside **Replacements** to toggle all replacements for the displayed
session, including future speech groups and copied text. The copy icon copies the
current transcript. The knob snaps on/off with a click; it cannot be dragged.
There are no playback controls or separate changes view.
Scrolling up or selecting text pauses following new content; **Jump to latest**
appears when there is newer text or you are away from the end. History opens
saved transcripts, with older content paged. Completed entries show text and
date/duration, without success-status lines. Notices are reserved for problems
that need attention, such as retrying a failed transcription or fixing a spelling.

Pause/Resume and Stop appear only inside the transcription pane during recording,
in a reserved location with fixed button sizes. Pause checkpoints captured audio
and flushes the current speech group without finishing the session; Resume
continues the same recording. Paused time is excluded from saved audio and elapsed
recording time. Stop also works while paused. The floating recording indicator
keeps its waveform and counter.

Press **Ctrl+Alt+Space** or Record to start, and press again to Stop. Stop saves
and finishes outstanding transcription, then copies the transcript with enabled
dictionary replacements. Turn Replacements off to view and copy unchanged ASR text. There is no simulated paste or destination
window tracking. Cancel retains audio for recovery and suppresses automatic copy.
A failed or empty transcription leaves the clipboard unchanged. Reopening GTK,
retrying failed work and reprocessing old audio never replay an automatic copy.

Silence stays in saved audio and is skipped only for inference. Speech is grouped
at natural pauses, normally around 30–60 seconds, with a 90-second input ceiling.
CPU Silero detection falls back visibly to bounded unfiltered chunks if its
verified model is unavailable. Reprocess can bypass silence filtering explicitly.

The service starts with Parakeet sleeping. Speech loads it in the background,
and one worker processes queued transcriptions serially. The default idle timeout
is five minutes. Preferences controls idle timeout and keeping the model loaded.
The bottom model-status row provides Load, Unload and Retry. Inference failures
preserve recording and available text.
Microphone, storage and capture-buffer failures stop recording visibly and retain
its saved prefix. A service restart recovers saved audio and pending work without
reopening the microphone.

Build dependencies on Ubuntu/Debian (the apt command installs system packages):

```bash
sudo apt install build-essential cmake pkg-config libgtk-3-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev libjson-glib-dev libxdo-dev
git submodule update --init third_party/cutlass vendor/llama.cpp
make desktop
make prepare-models
# Prepare paraketto-ultra-fp16.bin using the Parakeet Ultra instructions above.
./paraketto-service
```

`make prepare-models` explicitly prepares the pinned Silero asset; it does not
download the desktop's Ultra ASR weights. Convert those using the
[Parakeet Ultra instructions](#parakeet-ultra), or start the service with
`--model v2` or `--model v3` after preparing that preset's weights with the CLI.
`--weights /absolute/path/to/existing-weights.bin` selects a custom checkpoint.
Downloads never happen on
Stop. Python is used only for preparation. The service uses GIO,
JSON-GLib, GStreamer, vendored SQLite and CPU ggml, with no GTK or CUDA linkage.
ASR runs in one owned GPU process. No language model runtime or Python is needed
at runtime. See [VAD provenance](tests/vadfixtures/README.md).

For the foreground service above, run `./paraketto-desktop` in another terminal.
To use a user systemd service instead, stop the foreground service first, then run:

```bash
make install-service
./paraketto-desktop --install-desktop
./paraketto-desktop
```

To persist a different model choice, use
`uv run python scripts/install_service.py --model v3` instead of
`make install-service`. The installer also accepts `--weights FILE`.

The installer starts a user systemd service from this checkout, with no sudo.
Rerun it if the checkout moves. The desktop installer adds the launcher and GNOME
shortcut while preserving existing shortcuts; it does not enable desktop autostart.
Closing the window hides it. Explicit **Quit Paraketto** or
`./paraketto-desktop --quit` stops and saves its active recording. Stop/start the
service with `systemctl --user stop paraketto.service` and
`systemctl --user start paraketto.service`.

Session metadata, immutable transcript revisions, dictionaries, reversible edits
and Stop publications live in `~/.local/share/paraketto/sessions.sqlite3`.
Audio is stored as 30-second PCM blocks under `sessions/<id>/audio/`, with roughly
one-second synchronization checkpoints. Capture buffers, audio reads and transcript
paging are bounded; explicit full-text exports and clipboard copies materialize
the requested transcript.
A sudden power loss may lose an unsynchronized tail; recovery adopts readable
frames from an interrupted block. Legacy WAV/JSON history is imported idempotently
and labelled as legacy text, since earlier correction may already have changed it.
Explicit deletion removes the session's owned files. Preferences and logs remain
in this data directory. `$XDG_DATA_HOME` is respected.

The service accepts `--model v2|v3|ultra` with the same preset lookup as the CLI
and defaults to Ultra. It also accepts `--weights FILE`,
`--worker-binary FILE`, `--vad-model FILE`, `--data-dir DIRECTORY`, `--port PORT` and
`--load-timeout SECONDS` (120 by default). An explicit occupied port fails;
`--port 0` chooses a free port. Without a port argument, 8080 is preferred with
an automatic free-port fallback. The desktop discovers the service and never
starts or stops it. Use `--service-url URL` for an explicit endpoint or matching
`--data-dir DIRECTORY` arguments for isolated instances. The file chooser and
`--transcribe FILE` import existing WAV recordings with bounded ingestion.

CMake supports `-DPARAKETTO_BUILD_DESKTOP=ON` (includes the service) and
`-DPARAKETTO_BUILD_SERVICE=ON`. The service's ggml dependency is CPU-only.

Focused CPU checks:

```bash
make test-session-store test-capture test-segmenter test-asr-scheduling
make test-service test-sessions test-desktop test-desktop-lifecycle test-desktop-pane
make test-replacement-highlights
```

Desktop tests use a private Xvfb/D-Bus session and require `xvfb`; no test changes
the user's clipboard, focus or shortcuts. Service tests use isolated directories
and fake inference workers. Explicit opt-in checks are
`python3 tests/bench_sessions.py`,
`python3 tests/bench_pipeline.py` (GPU), and
`python3 tests/test_long_session.py` (62-minute real-time public-speech capture,
no microphone or GPU). GPU benchmarks check `nvidia-smi` before starting.

### Local service API

The persistent service binds only to loopback and rejects cross-origin browser
requests. It publishes its identity, port and PID in `<data-dir>/service.json`,
readable only by the current user. The desktop checks that identity before sending
requests. Shutdown uses systemd/signals. Standalone `paraketto.cuda --server` and
its existing web UI remain available separately.

| Endpoint | Behaviour |
|---|---|
| `GET /health`, `/status`, `/info` | Identity/API v1, ASR state, queue, active session, settings and known model metadata |
| `POST /transcribe` | Legacy multipart WAV transcription; bounded queue, raw transcription JSON |
| `POST /model/load`, `/model/unload`, `/model/release` | Legacy ASR lifecycle and recording reservations |
| `POST /models/asr/{load,unload}` | Explicit model lifecycle; unload returns 409 while in use |
| `PATCH /settings` | `idle_seconds` (1–86400), `keep_loaded` |
| `POST /sessions` | Start capture with UUID `request_id`; optional `silence_filter` |
| `POST /sessions/import` | Import absolute local WAV `path` with UUID `request_id` |
| `GET /sessions?limit=50&cursor=0` | Bounded history metadata and statistics |
| `GET /sessions/{id}` | Capture, ASR and source-timeline progress, generation and publication identity |
| `GET /sessions/{id}/updates?after=0&limit=50&generation=1` | Bounded changed segments/edits and next cursor; `reset` requests a refresh |
| `GET /sessions/{id}/levels` | Small elapsed-time/level response and captured/written/durable sample counts |
| `POST /sessions/{id}/pause`, `/resume` | Suspend/resume capture in the same session without publishing or copying |
| `POST /sessions/{id}/stop`, `/cancel` | Save/finish capture or cancel work while retaining audio |
| `POST /sessions/{id}/retry` | Retry failed `stage`: `all` or `asr` |
| `POST /sessions/{id}/reprocess` | New generation with `audio`, `silence_filter`, optional `dictionary_revision` |
| `GET /sessions/{id}/publication` | Immutable Stop snapshot and completion status |
| `GET /sessions/{id}/transcript?view=edited` | Explicit full text export: `edited`, `original` or `publication` |
| `GET /sessions/{id}/audio?start=0&end=16000` | Bounded WAV source range, offsets in 16 kHz samples; HTTP Range supported |
| `PATCH /sessions/{id}/replacements` | Global `enabled` flag for existing/future dictionary replacements; optional `expected_revision` rejects stale guarded writes |
| `GET /dictionary`, `PUT /dictionary` | Immutable revisions of `entries`; optional `expected_revision` protects full-list saves |
| `POST /dictionary/entries` | Atomically merge one canonical spelling and its aliases without replacing other entries |
| `PUT /dictionary/entries/{index}`, `DELETE /dictionary/entries/{index}` | Change/remove one entry; required `expected_revision` guards its list index |
| `POST /sessions/{id}/spelling` | Save a selected transcript spelling and alias with source/revision checks |
| `DELETE /sessions/{id}` | Cancel work, tombstone session and recoverably remove its owned files |

The legacy upload limit is 64 MiB, with four accepted transcriptions including
the active request; excess requests return 429. New session capture is block-based
and does not have that duration-dependent upload limit. Errors use JSON `error`.
Legacy recording reservations use UUID `lease_id`, expire after ten seconds and
can be renewed with `/model/load`; session capture manages its own activity.

### Dictionary replacements

The explicit dictionary maps aliases to canonical terms using Unicode-aware word
boundaries and configured case matching. Replacements do not cascade. Every edit
references the immutable original transcript. The Replacements pill applies or
removes all dictionary replacements in the session, without changing raw ASR.
A session keeps its dictionary revision. Saving a selected spelling updates the
selected section and the rules for subsequent speech groups without interrupting
recording; other existing sections remain unchanged. Reprocess with `audio: false`
to apply the latest dictionary throughout an existing transcript.

Rules are compiled once per cached revision into an Aho–Corasick matcher, with
an ASCII fast path and exact Unicode byte mappings. No dictionary parsing or
rule rebuilding occurs on a cache hit. [Dictionary usage, API and measurements](docs/dictionary.md).

There is no automatic LLM rewriting. S1-mini and the old synchronous corrector
have been removed. Upgrading an older session database retires old model edits
from the current transcript while preserving raw text, recordings, immutable Stop
publications and historical audit data. Schema v4 stores the global replacement
setting; old sessions with any disabled dictionary change migrate with replacements
off, preserving that opt-out. SQLite creates a backup before migration.

### Embedding as a library (CMake)

The Makefile is the canonical build for the standalone binaries. A separate CMake build produces a static library (`paraketto_lib`) for embedding the engine in other projects — see the header comment in `CMakeLists.txt` for options (`PARAKETTO_USE_CUBLAS`, `PARAKETTO_BUILD_LIB_ONLY`). The public API is `paraketto::Pipeline` in `src/paraketto.h`. Desktop/service targets are optional.

## HTTP API

- `GET /` — web UI (microphone recording + file upload; shows the running model/backend)
- `GET /health` — returns `{"status":"ok"}`
- `GET /info` — model name, `model_preset` (`v2`, `v3`, `ultra`, or `custom`), `model_architecture` (`v2` or `v3`), numeric architecture `version`, backend (CUTLASS FP16/cuBLAS FP16), weights path, device and startup timing
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
make test-model-selection test-model-selection-cli test-service-models
make test-cublas-fallback
```

## Project structure

```
src/paraketto_cuda.cpp    # CLI, startup, and HTTP server entry point
src/paraketto.h           # Public pipeline API, chunking, and greedy decode
src/model_defs.h          # Shared model constants, config, Weights struct
src/model_selection.h     # Presets, weight lookup and CPU architecture validation
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
src/service/             # Durable sessions, audio capture, VAD, ASR scheduling and dictionary edits
src/desktop/             # GTK session pane, history and recording pill
scripts/weight_format.py  # Shared FP16 tensor order and alignment
scripts/export_weights.py # ONNX → PRKT FP16 converter (V2 / V3)
scripts/export_ultra_weights.py # Safetensors → PRKT FP16 converter (Ultra)
```

## Acknowledgments

- **[Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)** by NVIDIA — the original ASR model ([V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) English, [V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) multilingual). This project is a from-scratch C++/CUDA reimplementation of the inference pipeline. Model weights are used under CC-BY-4.0, Copyright NVIDIA Corporation.
- **[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)** by Moondream — a post-trained V3 checkpoint, also CC-BY-4.0. The optional FP16 converter preserves our V3 inference layout.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA Templates for Linear Algebra Subroutines, used as a git submodule for FP16 GEMMs (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — CPU ggml VAD via git submodule (MIT License, Copyright 2023-2026 The ggml authors)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server (MIT License)

## References

- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [V3 paper](https://arxiv.org/abs/2509.14128) — Canary-1B-v2 & Parakeet-TDT-0.6B-v3
