# CUDA toolkit: newest /usr/local/cuda-* install, else nvcc from PATH.
# Override with CUDA_HOME=/path/to/cuda.
CUDA_LATEST  := $(lastword $(sort $(wildcard /usr/local/cuda-*)))
NVCC_IN_PATH := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME    ?= $(or $(CUDA_LATEST),$(patsubst %/bin/nvcc,%,$(NVCC_IN_PATH)),/usr/local/cuda)

# GPU arch: auto-detected from the installed GPU (e.g. 12.0 -> sm_120).
# Override with ARCH=sm_XX.
GPU_CC   := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH     ?= $(if $(GPU_CC),sm_$(GPU_CC),sm_120)

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
NVFLAGS  = -std=c++17 -O3 -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr

HF_BASE     = https://huggingface.co/localoptima/paraketto/resolve/main
WEIGHTS_DIR = $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/paraketto
WEIGHTS     = $(WEIGHTS_DIR)/paraketto-fp16.bin

.DEFAULT_GOAL := paraketto.cuda

.PHONY: bench-all bench-cuda bench-cublas bench-v3 weights weights-export download-data download-data-v3 download-weights check-weights repack clean

# Verify paraketto-fp16.bin is the expected format (PRKT v2)
check-weights: $(WEIGHTS)
	@v=$$(od -An -td4 -N4 -j4 $(WEIGHTS) | tr -d ' '); \
	if [ "$$v" != "2" ]; then \
		echo "ERROR: paraketto-fp16.bin is version $$v, expected 2. Run: uv run python scripts/repack_weights.py"; \
		exit 1; \
	fi

# Download benchmark data from HuggingFace
data/librispeech/manifest.json:
	@echo "Downloading benchmark data..."
	@wget -q --show-progress -O bench-data.tar.gz $(HF_BASE)/bench-data.tar.gz && \
		tar xzf bench-data.tar.gz && \
		rm bench-data.tar.gz && \
		echo "Downloaded $$(find data/ -name '*.wav' | wc -l) wav files"

data/german/manifest.json:
	@echo "Downloading V3 benchmark data..."
	@wget -q --show-progress -O bench-data-v3.tar.gz $(HF_BASE)/bench-data-v3.tar.gz && \
		tar xzf bench-data-v3.tar.gz && \
		rm bench-data-v3.tar.gz && \
		echo "Downloaded $$(find data/german data/italian data/french -name '*.wav' | wc -l) wav files"

download-data: data/librispeech/manifest.json
download-data-v3: data/german/manifest.json

# Download weights from HuggingFace
$(WEIGHTS):
	@mkdir -p $(WEIGHTS_DIR)
	@echo "Downloading paraketto-fp16.bin..."
	@wget -q --show-progress -O $@ $(HF_BASE)/paraketto-fp16.bin
	@echo "Downloaded $@ ($$(du -h $@ | cut -f1))"

download-weights: $(WEIGHTS)
weights: $(WEIGHTS)

BENCH_SEP = @printf '\n%s\n%s\n%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' '  $(1)'

bench-all: paraketto.cuda paraketto.cublas data/librispeech/manifest.json $(WEIGHTS) check-weights
	$(call BENCH_SEP,C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16)
	@uv run python tests/bench_native.py paraketto.cuda
	$(call BENCH_SEP,C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16)
	@uv run python tests/bench_native.py paraketto.cublas

bench-cuda: paraketto.cuda data/librispeech/manifest.json $(WEIGHTS) check-weights
	uv run python tests/bench_native.py paraketto.cuda

bench-cublas: paraketto.cublas data/librispeech/manifest.json $(WEIGHTS) check-weights
	uv run python tests/bench_native.py paraketto.cublas

bench-v3: paraketto.cuda data/german/manifest.json
	uv run python tests/test_v3_multilingual.py paraketto.cuda

# Re-generate weights from ONNX (only needed if export script changes)
weights-export: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h src/common.h
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -c $< -o $@

SHARED_HEADERS = src/paraketto.h src/model_selection.h src/common.h src/wav.h src/mel.h src/vocab.h src/server.h src/web_ui.h src/gpu_status.h

# Shared CUDA backend flags
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl
CUTLASS_INC   = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include

# Friendly error when the CUTLASS submodule hasn't been initialized
third_party/cutlass/include:
	@echo "ERROR: CUTLASS submodule missing. Run: git submodule update --init third_party/cutlass"; exit 1
src/weights.o: src/weights.cpp src/model_defs.h src/common.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@



CONFORMER_DEPS = src/paraketto_cuda.cpp src/conformer.cpp src/conformer.h src/model_defs.h src/kernels.h src/gemm.h $(SHARED_HEADERS)

# CUTLASS GEMM backend (default — no cuBLAS dependency, cudart only)
src/cutlass_gemm.o: src/cutlass_gemm.cu src/gemm.h src/kernels.h src/common.h | third_party/cutlass/include
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) -c $< -o $@

paraketto.cuda: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cutlass_gemm.o
	$(CXX) $(CUDA_CXXFLAGS) -DBACKEND_NAME='"CUTLASS FP16"' src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cutlass_gemm.o $(CUDA_LDFLAGS) -o $@

# cuBLAS GEMM backend (faster on some shapes, requires libcublas)
src/cublas_gemm.o: src/cublas_gemm.cu src/gemm.h src/kernels.h src/common.h
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -c $< -o $@

paraketto.cublas: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cublas_gemm.o
	$(CXX) $(CUDA_CXXFLAGS) -DBACKEND_NAME='"cuBLAS FP16"' src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cublas_gemm.o $(CUDA_LDFLAGS) -lcublas -lcublasLt -o $@

# Convert existing weight files to current format (run once after updating)
repack: $(WEIGHTS)
	uv run python scripts/repack_weights.py $(WEIGHTS)

bench_gemm: tests/bench_gemm.cu
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) tests/bench_gemm.cu -lcublas -lcublasLt -o $@

bench_splitk: tests/bench_splitk.cu
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) tests/bench_splitk.cu -lcublas -lcublasLt -o $@

bench_tiles: tests/bench_tiles.cu
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) tests/bench_tiles.cu -lcublas -lcublasLt -o $@

bench_ff2: tests/bench_ff2.cu
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) tests/bench_ff2.cu -lcublas -lcublasLt -o $@

.PHONY: test-cublas-fallback
test-cublas-fallback: scratch/test_cublas_fallback
	./scratch/test_cublas_fallback

scratch/test_cublas_fallback: tests/test_cublas_fallback.cpp src/cublas_gemm.o src/kernels.o src/gemm.h src/common.h
	mkdir -p scratch
	$(CXX) $(CUDA_CXXFLAGS) $< src/cublas_gemm.o src/kernels.o $(CUDA_LDFLAGS) -lcublas -lcublasLt -Wl,--wrap=cublasLtMatmulAlgoGetHeuristic -o $@

clean:
	rm -f paraketto.cuda paraketto.cublas
	rm -f bench_ff2 bench_gemm bench_splitk bench_tiles
	rm -f src/*.o
	rm -f scratch/test_cublas_fallback scratch/test_desktop
	rm -f paraketto-desktop paraketto-service

# Optional native Linux app. The inference builds do not need these libraries.
DESKTOP_PKGS = gtk+-3.0 gio-unix-2.0 gstreamer-app-1.0 json-glib-1.0 x11
DESKTOP_SOURCES = src/desktop/app.cpp src/desktop/audio.cpp src/desktop/backend.cpp src/desktop/dictionary_view.cpp src/desktop/history.cpp src/desktop/input.cpp
APP_HEADERS = src/app_common.h src/model_settings.h src/model_selection.h src/service/api.h
DESKTOP_HEADERS = $(wildcard src/desktop/*.h) $(APP_HEADERS) src/wav.h
SERVICE_CORE = src/service/worker.cpp src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp src/service/capture.cpp src/service/segmenter.cpp src/service/sessions.cpp src/service/vad/vad_ggml.cpp
SERVICE_SOURCES = src/service/main.cpp $(SERVICE_CORE)
SERVICE_PKGS = gio-unix-2.0 json-glib-1.0 gstreamer-app-1.0
SERVICE_HEADERS = $(wildcard src/service/*.h src/service/vad/*.h) $(APP_HEADERS)
GGML_SERVICE = scratch/ggml-service
GGML_SERVICE_LIBS = $(GGML_SERVICE)/ggml/src/libggml.a $(GGML_SERVICE)/ggml/src/libggml-cpu.a $(GGML_SERVICE)/ggml/src/libggml-base.a
SERVICE_LIBS = -Wl,--start-group $(GGML_SERVICE_LIBS) -Wl,--end-group -ldl
SERVICE_INCLUDES = -Ithird_party -Ithird_party/sqlite -Isrc -Ivendor/llama.cpp/ggml/include

$(GGML_SERVICE)/.stamp: vendor/llama.cpp/ggml/include/ggml.h
	cmake -S vendor/llama.cpp -B $(GGML_SERVICE) -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
		-DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_COMMON=OFF \
		-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
	cmake --build $(GGML_SERVICE) --target ggml --parallel 6
	touch $@

scratch/sqlite3.o: third_party/sqlite/sqlite3.c third_party/sqlite/sqlite3.h
	mkdir -p scratch
	$(CC) -O2 -DSQLITE_THREADSAFE=1 -DSQLITE_DEFAULT_FOREIGN_KEYS=1 -c $< -o $@

.PHONY: prepare-models
prepare-models:
	uv run --no-project python scripts/prepare_vad.py

paraketto-service: $(SERVICE_SOURCES) $(SERVICE_HEADERS) src/web_ui.h scratch/sqlite3.o $(GGML_SERVICE)/.stamp
	pkg-config --exists $(SERVICE_PKGS)
	$(CXX) -std=c++17 -O2 -Wall -Wextra $(SERVICE_INCLUDES) \
		$(SERVICE_SOURCES) scratch/sqlite3.o $$(pkg-config --cflags --libs $(SERVICE_PKGS)) $(SERVICE_LIBS) -pthread -o $@

.PHONY: desktop
desktop: paraketto.cuda paraketto-service paraketto-desktop

.PHONY: install-service
install-service: paraketto-service
	uv run --no-project python scripts/install_service.py

paraketto-desktop: $(DESKTOP_SOURCES) $(DESKTOP_HEADERS)
	pkg-config --exists $(DESKTOP_PKGS)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-deprecated-declarations -Ithird_party -Isrc \
		$(DESKTOP_SOURCES) $$(pkg-config --cflags --libs $(DESKTOP_PKGS)) -lxdo -pthread -o $@

.PHONY: test-desktop
test-desktop: scratch/test_desktop
	sh tests/run_desktop_tests.sh

.PHONY: test-replacement-highlights
test-replacement-highlights: scratch/test_replacement_highlights
	./scratch/test_replacement_highlights

scratch/test_replacement_highlights: tests/test_replacement_highlights.cpp src/desktop/replacement_highlights.h
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Isrc $< -o $@

.PHONY: test-desktop-lifecycle
test-desktop-lifecycle: paraketto-desktop paraketto-service
	sh tests/run_desktop_tests.sh uv run --no-project python tests/test_desktop_lifecycle.py

.PHONY: test-desktop-pane
test-desktop-pane: scratch/test_desktop_pane paraketto-service
	sh tests/run_desktop_tests.sh uv run --no-project python tests/test_desktop_pane.py

scratch/test_desktop_pane: tests/test_desktop_pane.cpp $(DESKTOP_SOURCES) $(DESKTOP_HEADERS) src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp $(SERVICE_HEADERS) scratch/sqlite3.o
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-deprecated-declarations $(SERVICE_INCLUDES) $< \
		$(filter-out src/desktop/app.cpp,$(DESKTOP_SOURCES)) src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp \
		scratch/sqlite3.o $$(pkg-config --cflags --libs $(DESKTOP_PKGS)) -lxdo -pthread -o $@

.PHONY: test-service
test-service: paraketto-service
	uv run --no-project python tests/test_service.py

.PHONY: test-model-selection test-model-selection-cli test-service-models
test-model-selection: scratch/test_model_selection
	./scratch/test_model_selection

scratch/test_model_selection: tests/test_model_selection.cpp src/model_selection.h
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -Isrc $< -o $@

test-model-selection-cli: paraketto.cuda
	uv run --no-project python tests/test_model_selection_cli.py

test-service-models: paraketto-service
	uv run --no-project python tests/test_service_models.py

.PHONY: test-sessions test-session-store test-capture test-segmenter test-asr-scheduling
test-asr-scheduling: scratch/test_asr_scheduling
	./scratch/test_asr_scheduling tests/desktop_fixture_server.py

scratch/test_asr_scheduling: tests/test_asr_scheduling.cpp src/service/main.cpp src/service/worker.cpp $(SERVICE_HEADERS)
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -ffunction-sections -fdata-sections $(SERVICE_INCLUDES) $< src/service/worker.cpp $$(pkg-config --cflags --libs $(SERVICE_PKGS)) -pthread -Wl,--gc-sections -o $@

test-sessions: paraketto-service
	uv run --no-project python tests/test_sessions.py

test-session-store: scratch/test_session_store
	./scratch/test_session_store

.PHONY: bench-dictionary
bench-dictionary: scratch/bench_dictionary
	./scratch/bench_dictionary

scratch/bench_dictionary: tests/bench_dictionary.cpp src/service/dictionary.cpp src/service/dictionary.h src/service/edits.h src/app_common.h
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc tests/bench_dictionary.cpp src/service/dictionary.cpp $$(pkg-config --cflags --libs json-glib-1.0) -o $@

scratch/test_session_store: tests/test_session_store.cpp src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp $(SERVICE_HEADERS) scratch/sqlite3.o
	$(CXX) -std=c++17 -O2 -Wall -Wextra $(SERVICE_INCLUDES) $< src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp scratch/sqlite3.o $$(pkg-config --cflags --libs gio-unix-2.0 json-glib-1.0) -pthread -o $@

test-capture: scratch/test_capture
	./scratch/test_capture

scratch/test_capture: tests/test_capture.cpp src/service/capture.cpp src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp $(SERVICE_HEADERS) scratch/sqlite3.o
	$(CXX) -std=c++17 -O2 -Wall -Wextra $(SERVICE_INCLUDES) $< src/service/capture.cpp src/service/session_store.cpp src/service/dictionary.cpp src/service/edits.cpp scratch/sqlite3.o $$(pkg-config --cflags --libs $(SERVICE_PKGS)) -pthread -o $@

test-segmenter: scratch/test_segmenter
	./scratch/test_segmenter

scratch/test_segmenter: tests/test_segmenter.cpp src/service/segmenter.cpp src/service/vad/vad_ggml.cpp $(SERVICE_HEADERS) $(GGML_SERVICE)/.stamp
	$(CXX) -std=c++17 -O2 -Wall -Wextra $(SERVICE_INCLUDES) $< src/service/segmenter.cpp src/service/vad/vad_ggml.cpp $$(pkg-config --cflags --libs $(SERVICE_PKGS)) $(SERVICE_LIBS) -pthread -o $@

scratch/test_desktop: tests/test_desktop.cpp $(filter-out src/desktop/app.cpp,$(DESKTOP_SOURCES)) src/service/worker.cpp src/service/worker.h $(DESKTOP_HEADERS)
	mkdir -p scratch
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-deprecated-declarations -Ithird_party -Isrc \
		$< $(filter-out src/desktop/app.cpp,$(DESKTOP_SOURCES)) src/service/worker.cpp \
		$$(pkg-config --cflags --libs $(DESKTOP_PKGS)) -lxdo -pthread -o $@
