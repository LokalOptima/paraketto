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

.PHONY: bench-all bench-cuda bench-cublas bench-corrector bench-v3 weights weights-export download-data download-data-v3 download-weights check-weights repack clean llama-libs

# ---------------------------------------------------------------------------
# LLM corrector (opt-in via WITH_CORRECTOR=1)
# ---------------------------------------------------------------------------
# Builds llama.cpp from the vendored submodule as static libraries, then links
# them into paraketto. Only the CUDA backend is built; everything else is off.

LLAMA_SRC    = vendor/llama.cpp
LLAMA_BUILD  = $(LLAMA_SRC)/build
LLAMA_LIBDIR = $(LLAMA_BUILD)/src $(LLAMA_BUILD)/ggml/src $(LLAMA_BUILD)/ggml/src/ggml-cuda $(LLAMA_BUILD)/ggml/src/ggml-cpu
LLAMA_INC    = -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include
LLAMA_STAMP  = $(LLAMA_BUILD)/.stamp

# Static lib build (one-time, cached)
$(LLAMA_STAMP):
	cmake -B $(LLAMA_BUILD) $(LLAMA_SRC) \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_CUDA=ON -DCUDA_ARCHITECTURES=120 \
		-DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_RPC=OFF \
		-DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=ON \
		-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_BUILD_COMMON=OFF \
		-DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
		-DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
		-DCMAKE_CUDA_FLAGS="-Xcompiler -ffunction-sections -Xcompiler -fdata-sections"
	cmake --build $(LLAMA_BUILD) --target llama --parallel
	touch $@

llama-libs: $(LLAMA_STAMP)

ifdef WITH_CORRECTOR
  CORRECTOR_OBJ     = src/corrector.o
  CORRECTOR_CFLAGS  = -DWITH_CORRECTOR $(LLAMA_INC)
  CORRECTOR_LDFLAGS = $(foreach d,$(LLAMA_LIBDIR),-L$(d)) \
                      -Wl,--whole-archive -lggml-cuda -Wl,--no-whole-archive \
                      -lllama -lggml -lggml-base -lggml-cpu \
                      -lcudart -lcublas -lcublasLt -lcuda -lgomp \
                      -Wl,--gc-sections
  CORRECTOR_DEP     = $(LLAMA_STAMP)
endif

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

bench-corrector: paraketto.cuda data/librispeech/manifest.json $(WEIGHTS)
	uv run python tests/bench_corrector.py paraketto.cuda

bench-v3: paraketto.cuda data/german/manifest.json
	uv run python tests/test_v3_multilingual.py paraketto.cuda

# Re-generate weights from ONNX (only needed if export script changes)
weights-export: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h src/common.h
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -c $< -o $@

SHARED_HEADERS = src/paraketto.h src/common.h src/wav.h src/mel.h src/vocab.h src/server.h

# Shared CUDA backend flags
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread
CUTLASS_INC   = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include

# Friendly error when the CUTLASS submodule hasn't been initialized
third_party/cutlass/include:
	@echo "ERROR: CUTLASS submodule missing. Run: git submodule update --init third_party/cutlass"; exit 1
src/weights.o: src/weights.cpp src/model_defs.h src/common.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

ifdef WITH_CORRECTOR
src/corrector.o: src/corrector.cpp src/corrector.h $(CORRECTOR_DEP)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) -ffunction-sections -fdata-sections -c $< -o $@
endif

CONFORMER_DEPS = src/paraketto_cuda.cpp src/conformer.cpp src/conformer.h src/model_defs.h src/kernels.h src/gemm.h $(SHARED_HEADERS)

# CUTLASS GEMM backend (default — no cuBLAS dependency, cudart only)
src/cutlass_gemm.o: src/cutlass_gemm.cu src/gemm.h src/kernels.h src/common.h | third_party/cutlass/include
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $(CUTLASS_INC) -c $< -o $@

paraketto.cuda: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cutlass_gemm.o $(CORRECTOR_OBJ)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) -DBACKEND_NAME='"CUTLASS FP16"' src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cutlass_gemm.o $(CORRECTOR_OBJ) $(CUDA_LDFLAGS) $(CORRECTOR_LDFLAGS) -o $@

# cuBLAS GEMM backend (faster on some shapes, requires libcublas)
src/cublas_gemm.o: src/cublas_gemm.cu src/gemm.h src/kernels.h src/common.h
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) -c $< -o $@

paraketto.cublas: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cublas_gemm.o $(CORRECTOR_OBJ)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) -DBACKEND_NAME='"cuBLAS FP16"' src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cublas_gemm.o $(CORRECTOR_OBJ) $(CUDA_LDFLAGS) -lcublas -lcublasLt $(CORRECTOR_LDFLAGS) -o $@

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
	rm -f scratch/test_cublas_fallback
