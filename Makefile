CUDA_HOME      ?= /usr/local/cuda-13.1

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
NVFLAGS  = -std=c++17 -O3 -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr

HF_BASE     = https://huggingface.co/localoptima/paraketto/resolve/main
WEIGHTS_DIR = $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/paraketto
WEIGHTS     = $(WEIGHTS_DIR)/paraketto-fp16.bin
WEIGHTS_FP8 = $(WEIGHTS_DIR)/paraketto-fp8.bin

.PHONY: bench-all bench-cuda bench-cublas bench-fp8 bench-corrector weights weights-fp8 download-data download-weights check-weights check-weights-fp8 clean llama-libs

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

# Verify paraketto-fp8.bin is the expected format (PRKTFP8 v1)
check-weights-fp8: $(WEIGHTS_FP8)
	@m=$$(od -An -tx1 -N7 $(WEIGHTS_FP8) | tr -d ' '); \
	v=$$(od -An -td4 -N4 -j8 $(WEIGHTS_FP8) | tr -d ' '); \
	if [ "$$m" != "50524b54465038" ] || [ "$$v" != "1" ]; then \
		echo "ERROR: paraketto-fp8.bin has invalid header (magic=$$m, version=$$v). Re-run to regenerate."; \
		exit 1; \
	fi

# Download benchmark data from HuggingFace
data/librispeech/manifest.json:
	@echo "Downloading benchmark data..."
	@wget -q --show-progress -O bench-data.tar.gz $(HF_BASE)/bench-data.tar.gz && \
		tar xzf bench-data.tar.gz && \
		rm bench-data.tar.gz && \
		echo "Downloaded $$(find data/ -name '*.wav' | wc -l) wav files"

download-data: data/librispeech/manifest.json

# Download weights from HuggingFace
$(WEIGHTS):
	@mkdir -p $(WEIGHTS_DIR)
	@echo "Downloading paraketto-fp16.bin..."
	@wget -q --show-progress -O $@ $(HF_BASE)/paraketto-fp16.bin
	@echo "Downloaded $@ ($$(du -h $@ | cut -f1))"

$(WEIGHTS_FP8):
	@mkdir -p $(WEIGHTS_DIR)
	@echo "Downloading paraketto-fp8.bin..."
	@wget -q --show-progress -O $@ $(HF_BASE)/paraketto-fp8.bin
	@echo "Downloaded $@ ($$(du -h $@ | cut -f1))"

download-weights: $(WEIGHTS)
weights: $(WEIGHTS)
weights-fp8: $(WEIGHTS_FP8)

BENCH_SEP = @printf '\n%s\n%s\n%s\n' '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' '  $(1)'

bench-all: paraketto.cuda paraketto.cublas paraketto.fp8 data/librispeech/manifest.json $(WEIGHTS) check-weights
	$(call BENCH_SEP,C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16)
	@uv run python tests/bench_native.py paraketto.cuda
	$(call BENCH_SEP,C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16)
	@uv run python tests/bench_native.py paraketto.cublas
	$(call BENCH_SEP,C++ FP8  · paraketto_cuda.cpp + cublasLt FP8)
	@uv run python tests/bench_native.py paraketto.fp8

bench-cuda: paraketto.cuda data/librispeech/manifest.json $(WEIGHTS) check-weights
	uv run python tests/bench_native.py paraketto.cuda

bench-cublas: paraketto.cublas data/librispeech/manifest.json $(WEIGHTS) check-weights
	uv run python tests/bench_native.py paraketto.cublas

bench-fp8: paraketto.fp8 data/librispeech/manifest.json $(WEIGHTS_FP8) check-weights-fp8
	uv run python tests/bench_native.py paraketto.fp8

bench-corrector: paraketto.fp8 data/librispeech/manifest.json $(WEIGHTS_FP8)
	uv run python tests/bench_corrector.py paraketto.fp8

bench-v3: paraketto.fp8
	uv run python tests/test_v3_multilingual.py paraketto.fp8

# Re-generate weights from ONNX (only needed if export script changes)
weights-export: scripts/export_weights.py
	uv run python scripts/export_weights.py

# C++ / CUDA build
src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 -c $< -o $@

SHARED_HEADERS = src/common.h src/wav.h src/mel.h src/vocab.h src/server.h

# Shared CUDA backend flags
CUDA_CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -Wno-deprecated-declarations -I$(CUDA_HOME)/include -Ithird_party -Isrc
CUDA_LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread
CUTLASS_INC   = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include
src/weights.o: src/weights.cpp src/conformer.h src/common.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

ifdef WITH_CORRECTOR
src/corrector.o: src/corrector.cpp src/corrector.h $(CORRECTOR_DEP)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) -ffunction-sections -fdata-sections -c $< -o $@
endif

CONFORMER_DEPS = src/paraketto_cuda.cpp src/conformer.cpp src/conformer.h src/kernels.h src/gemm.h $(SHARED_HEADERS)

# CUTLASS GEMM backend (default — no cuBLAS dependency, cudart only)
src/cutlass_gemm.o: src/cutlass_gemm.cu src/cutlass_gemm.h src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) -c $< -o $@

paraketto.cuda: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cutlass_gemm.o src/cutlass_gemm.h $(CORRECTOR_OBJ)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cutlass_gemm.o $(CORRECTOR_OBJ) $(CUDA_LDFLAGS) $(CORRECTOR_LDFLAGS) -o $@

# cuBLAS GEMM backend (faster on some shapes, requires libcublas)
src/cublas_gemm.o: src/cublas_gemm.cu src/gemm.h src/kernels.h
	$(NVCC) $(NVFLAGS) -arch=sm_120 -c $< -o $@

paraketto.cublas: $(CONFORMER_DEPS) src/weights.o src/kernels.o src/cublas_gemm.o $(CORRECTOR_OBJ)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) src/paraketto_cuda.cpp src/conformer.cpp src/weights.o src/kernels.o src/cublas_gemm.o $(CORRECTOR_OBJ) $(CUDA_LDFLAGS) -lcublas -lcublasLt $(CORRECTOR_LDFLAGS) -o $@

# FP8 cublasLt backend
src/kernels_fp8.o: src/kernels_fp8.cu src/kernels_fp8.h
	$(NVCC) $(NVFLAGS) -arch=sm_120a -c $< -o $@

src/conformer_fp8.o: src/conformer_fp8.cpp src/conformer_fp8.h src/conformer.h src/kernels.h src/kernels_fp8.h
	$(CXX) $(CUDA_CXXFLAGS) -I$(CUDA_HOME)/include -c $< -o $@

paraketto.fp8: src/paraketto_cuda.cpp src/conformer_fp8.h src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o $(SHARED_HEADERS) $(CORRECTOR_OBJ)
	$(CXX) $(CUDA_CXXFLAGS) $(CORRECTOR_CFLAGS) -include src/conformer_fp8.h src/paraketto_cuda.cpp src/conformer_fp8.o src/weights.o src/kernels.o src/kernels_fp8.o $(CORRECTOR_OBJ) $(CUDA_LDFLAGS) -lcublas -lcublasLt $(CORRECTOR_LDFLAGS) -o $@

# (paraketto-fp8.bin generation removed — paraketto.fp8 auto-downloads from HF)

# Convert existing weight files to current format (run once after updating)
repack: $(WEIGHTS)
	uv run python scripts/repack_weights.py

bench_gemm: tests/bench_gemm.cu
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) tests/bench_gemm.cu -lcublas -lcublasLt -o $@

bench_splitk: tests/bench_splitk.cu
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) tests/bench_splitk.cu -lcublas -lcublasLt -o $@

bench_tiles: tests/bench_tiles.cu
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) tests/bench_tiles.cu -lcublas -lcublasLt -o $@

bench_ff2: tests/bench_ff2.cu
	$(NVCC) $(NVFLAGS) -arch=sm_120 $(CUTLASS_INC) tests/bench_ff2.cu -lcublas -lcublasLt -o $@

clean:
	rm -f paraketto.cuda paraketto.cublas paraketto.fp8 src/kernels.o src/kernels_fp8.o src/cutlass_gemm.o src/cublas_gemm.o src/weights.o src/conformer_fp8.o src/corrector.o
