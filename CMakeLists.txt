# CMake build for embedding paraketto as a static library (paraketto_lib)
# in other projects. The Makefile is the canonical build for the standalone
# binaries (paraketto.cuda / paraketto.cublas) and the only
# one that supports the LLM corrector (WITH_CORRECTOR=1).
#
# Backend selection mirrors the Makefile: CUTLASS FP16 by default,
# PARAKETTO_USE_CUBLAS for cuBLAS FP16.

cmake_minimum_required(VERSION 3.24)

# Must be set before project() enables CUDA, or compiler detection caches
# its own default (sm_75) and the GPU's real arch is never used.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(paraketto LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

option(PARAKETTO_USE_CUBLAS "Use cuBLAS instead of CUTLASS for FP16" OFF)
option(PARAKETTO_BUILD_LIB_ONLY "Only build the library, skip the executable" OFF)

# --- Library sources ---
set(LIB_SOURCES
    src/weights.cpp
    src/conformer.cpp
    src/kernels.cu
)

if(PARAKETTO_USE_CUBLAS)
    list(APPEND LIB_SOURCES src/cublas_gemm.cu)
else()
    list(APPEND LIB_SOURCES src/cutlass_gemm.cu)
endif()

add_library(paraketto_lib STATIC ${LIB_SOURCES})

target_include_directories(paraketto_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(paraketto_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_options(paraketto_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

target_link_libraries(paraketto_lib PUBLIC CUDA::cudart)
target_link_libraries(paraketto_lib PRIVATE pthread)

if(PARAKETTO_USE_CUBLAS)
    target_link_libraries(paraketto_lib PUBLIC CUDA::cublas CUDA::cublasLt)
    set(PARAKETTO_BACKEND_NAME "cuBLAS FP16")
else()
    set(PARAKETTO_BACKEND_NAME "CUTLASS FP16")
endif()

# --- Executable (optional) ---
if(NOT PARAKETTO_BUILD_LIB_ONLY)
    add_executable(paraketto src/paraketto_cuda.cpp)
    target_link_libraries(paraketto paraketto_lib)
    target_compile_definitions(paraketto PRIVATE
        BACKEND_NAME="${PARAKETTO_BACKEND_NAME}"
    )
    target_include_directories(paraketto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
endif()
