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
option(PARAKETTO_BUILD_DESKTOP "Build the optional native Linux dictation app" OFF)
option(PARAKETTO_BUILD_SERVICE "Build the persistent local transcription service" OFF)

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
    target_link_libraries(paraketto paraketto_lib ${CMAKE_DL_LIBS})
    target_compile_definitions(paraketto PRIVATE
        BACKEND_NAME="${PARAKETTO_BACKEND_NAME}"
    )
    target_include_directories(paraketto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
endif()

if(PARAKETTO_BUILD_DESKTOP OR PARAKETTO_BUILD_SERVICE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SERVICE REQUIRED IMPORTED_TARGET gio-unix-2.0 json-glib-1.0)
    add_executable(paraketto-service src/service/main.cpp src/service/worker.cpp)
    target_include_directories(paraketto-service PRIVATE src third_party)
    target_link_libraries(paraketto-service PRIVATE PkgConfig::SERVICE pthread)
endif()

if(PARAKETTO_BUILD_DESKTOP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DESKTOP REQUIRED IMPORTED_TARGET gtk+-3.0 gio-unix-2.0 ibus-1.0 gstreamer-app-1.0 json-glib-1.0 x11)
    find_library(XDO_LIBRARY xdo REQUIRED)
    add_executable(paraketto-desktop
        src/desktop/app.cpp src/desktop/audio.cpp src/desktop/backend.cpp
        src/desktop/history.cpp src/desktop/input.cpp)
    target_include_directories(paraketto-desktop PRIVATE src third_party)
    target_link_libraries(paraketto-desktop PRIVATE PkgConfig::DESKTOP ${XDO_LIBRARY} pthread)
endif()
