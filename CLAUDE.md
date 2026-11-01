# paraketto — agent notes

C++/CUDA speech-to-text engine for NVIDIA Parakeet TDT 0.6B (V2 English, V3 multilingual)
and Moondream Parakeet Ultra (V3 architecture).
No Python at runtime; Python (via uv) is only for benchmarks and weight tooling.

## Build

- Canonical build is the **Makefile**: `make paraketto.cuda` (CUTLASS FP16),
  `make paraketto.cublas` (cuBLAS FP16). Bare `make` builds CUTLASS FP16.
- GPU arch is auto-detected from `nvidia-smi`; override with `ARCH=sm_XX`.
- Needs the `third_party/cutlass` submodule (`git submodule update --init third_party/cutlass`).
- The `fp8-ultra` branch preserves FP8 inference and experiments; main is FP16 only.
- `CMakeLists.txt` is a secondary build for embedding `paraketto_lib` in other
  projects and optional desktop/service targets. Keep it in sync when adding sources.
- Long-form service uses vendored SQLite and CPU ggml from `vendor/llama.cpp`.
  `make prepare-models` explicitly prepares the pinned Silero VAD asset.
  Output is ASR plus reversible dictionary replacements; there is no LLM cleanup.
  Desktop defaults to Transcription with History and Dictionary tabs, global replacement
  pill, inline hover highlights, selection-to-spelling popover and capture
  Pause/Resume/Stop; no playback UI. Dictionary matchers are cached by immutable
  revision; see `docs/dictionary.md` for correction/API semantics and CPU timings.
  Dictionary management lives in `src/desktop/dictionary_view.*`: searchable,
  paged entries with individual Add/Change/Remove actions and guarded writes.
  Keep the replacement control's pill/knob appearance; clicking snaps it fully
  on/off, without dragging. Keep routine save/copy/complete
  confirmations out of the UI; show notices only when the user needs to act.
- Session validation: `make test-session-store test-capture test-segmenter
  test-asr-scheduling test-sessions`; GPU integration is opt-in via `tests/bench_sessions.py`.

## GPU etiquette

Other agents may be using the GPU. Run `nvidia-smi` before benchmarks or inference runs.
Compilation is CPU-only and always safe.

## Weights & data

- CLI and service accept `--model v2|v3|ultra`; shared selection lives in
  `src/model_selection.h`. CLI defaults to V2, service to Ultra. Lookup prefers
  the preset filename beside the executable, then the XDG user cache.
  `--weights` overrides the path; without an explicit `--model` it means custom
  weights with architecture auto-detection. Explicit presets verify the header.
- CLI V2/V3 weights auto-download to `~/.cache/paraketto/` (HF repo `localoptima/paraketto`).
  Ultra requires `scripts/export_ultra_weights.py`; service loads never download.
  Binary formats are versioned — `make check-weights` validates the default V2 header;
  `scripts/repack_weights.py` migrates old files.
- Benchmark data: `make download-data` (English) / `make download-data-v3` (de/it/fr FLEURS)
  into `data/` (gitignored).

## Benchmarks / tests

- Benchmark harnesses: `make bench-cuda|bench-cublas|bench-v3`.
- Ultra converter checks: `uv run python -m unittest discover -s tests -p test_ultra_conversion.py`.
- Selector checks: `make test-model-selection test-model-selection-cli test-service-models`.
- Forced cuBLAS fallback check: `make test-cublas-fallback`.
- Regression testing with statistical confidence: `tests/bench_regression.py`
  (`--save baseline.json` then `--compare`; baselines are machine-specific, gitignored).
- WER numbers in README come from these harnesses — update README tables when they change.

## Conventions

- `scratch/` is gitignored working space for one-off experiments and notes.
- `blog.md` is the development worklog — significant changes get an entry (see /commit-and-blog).
- Commit completed, validated work in coherent increments before reporting it
  finished; include required new files, tests and documentation. Push only when requested.
- `scripts/weight_format.py` defines the FP16 tensor order for both exporters
  and the legacy repacker; keep it consistent with `src/weights.cpp`.
