PARAKEET_MODEL ?= nemo-parakeet-tdt-0.6b-v2
VENV_PKGS       = .venv/lib/python3.13/site-packages
export LD_LIBRARY_PATH := $(VENV_PKGS)/tensorrt_libs:$(VENV_PKGS)/onnxruntime/capi:$(LD_LIBRARY_PATH)

.PHONY: run bench test

run: src/parakeet-transcribe.py
	arecord -f S16_LE -r 16000 -c 1 -t raw - | \
	  ORT_LOG_LEVEL=3 uv run src/parakeet-transcribe.py --model $(PARAKEET_MODEL) --cuda

bench:
	uv run python tests/bench_rtfx.py

test:
	uv run pytest tests/ -v
