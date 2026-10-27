# paraketto — agent notes

C++/CUDA speech-to-text engine for NVIDIA Parakeet TDT 0.6B (V2 English, V3 multilingual).
No Python at runtime; Python (via uv) is only for benchmarks and weight tooling.

## Build

- Canonical build is the **Makefile**: `make paraketto.cuda` (CUTLASS FP16),
  `make paraketto.cublas` (cuBLAS FP16). Bare `make` builds CUTLASS FP16.
- GPU arch is auto-detected from `nvidia-smi`; override with `ARCH=sm_XX`.
- Needs the `third_party/cutlass` submodule (`git submodule update --init third_party/cutlass`).
- The `fp8-ultra` branch preserves FP8 inference and experiments; main is FP16 only.
- `CMakeLists.txt` is a secondary build for embedding `paraketto_lib` in other
  projects; it does not support the LLM corrector. Keep it in sync when adding sources.
- LLM corrector is opt-in: `make paraketto.cuda WITH_CORRECTOR=1` (needs `vendor/llama.cpp` submodule).

## GPU etiquette

Other agents may be using the GPU. Run `nvidia-smi` before benchmarks or inference runs.
Compilation is CPU-only and always safe.

## Weights & data

- Weights auto-download to `~/.cache/paraketto/` (HF repo `localoptima/paraketto`).
  Binary formats are versioned — `make check-weights` validates the default V2 header;
  `scripts/repack_weights.py` migrates old files.
- Benchmark data: `make download-data` (English) / `make download-data-v3` (de/it/fr FLEURS)
  into `data/` (gitignored).

## Benchmarks / tests

- Benchmark harnesses: `make bench-cuda|bench-cublas|bench-v3`.
- Ultra converter checks: `uv run python -m unittest discover -s tests -p test_ultra_conversion.py`.
- Forced cuBLAS fallback check: `make test-cublas-fallback`.
- Regression testing with statistical confidence: `tests/bench_regression.py`
  (`--save baseline.json` then `--compare`; baselines are machine-specific, gitignored).
- WER numbers in README come from these harnesses — update README tables when they change.

## Conventions

- `scratch/` is gitignored working space for one-off experiments and notes.
- `blog.md` is the development worklog — significant changes get an entry (see /commit-and-blog).
- `scripts/weight_format.py` defines the FP16 tensor order for both exporters
  and the legacy repacker; keep it consistent with `src/weights.cpp`.
