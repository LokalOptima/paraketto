cmake_minimum_required(VERSION 3.24)
project(paraketto LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

option(PARAKETTO_FP8 "Build FP8 variant (requires Blackwell sm_120a)" ON)
option(PARAKETTO_USE_CUBLAS "Use cuBLAS instead of CUTLASS for FP16 (ignored with FP8)" OFF)
option(PARAKETTO_BUILD_LIB_ONLY "Only build the library, skip the executable" OFF)

# --- Library sources ---
set(LIB_SOURCES
    src/weights.cpp
    src/kernels.cu
)

if(PARAKETTO_FP8)
    list(APPEND LIB_SOURCES src/conformer_fp8.cpp src/kernels_fp8.cu src/cutlass_gemm.cu src/cutlass_gemm_fp8.cu)
elseif(PARAKETTO_USE_CUBLAS)
    list(APPEND LIB_SOURCES src/conformer.cpp src/cublas_gemm.cu)
else()
    list(APPEND LIB_SOURCES src/conformer.cpp src/cutlass_gemm.cu)
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

if(PARAKETTO_FP8)
    target_compile_definitions(paraketto_lib PUBLIC PARAKETTO_FP8=1)
    target_link_libraries(paraketto_lib PUBLIC CUDA::cublas CUDA::cublasLt)
elseif(PARAKETTO_USE_CUBLAS)
    target_link_libraries(paraketto_lib PUBLIC CUDA::cublas CUDA::cublasLt)
endif()

# --- Executable (optional) ---
if(NOT PARAKETTO_BUILD_LIB_ONLY)
    add_executable(paraketto src/paraketto_cuda.cpp)
    target_link_libraries(paraketto paraketto_lib)
    target_include_directories(paraketto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
endif()
