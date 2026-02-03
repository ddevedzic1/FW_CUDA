cmake_minimum_required(VERSION 3.18)
project(fw_cuda LANGUAGES CXX CUDA)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CUDA Standard
# =============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# Build type (default: Release)
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# Compiler Flags
# =============================================================================
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# =============================================================================
# Output Directories (set BEFORE defining targets)
# =============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# =============================================================================
# Include Directories
# =============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# =============================================================================
# Source Files - Core Library (CPU only)
# =============================================================================
set(CPU_SOURCES
    src/utils.cpp
    src/graph_generator.cpp
    src/fw_baseline_cpu.cpp
)

# =============================================================================
# Source Files - CUDA Library
# =============================================================================
set(CUDA_SOURCES
    src/cuda_utils.cu
    src/timer.cu
    src/gpu_memory.cu
    src/fw_baseline_gpu.cu
    src/fw_tiling_gpu.cu
    src/algorithm_runner.cu
)

# =============================================================================
# Static Library - CPU Algorithms
# =============================================================================
add_library(fw_cpu STATIC ${CPU_SOURCES})

# =============================================================================
# Static Library - CUDA Components
# =============================================================================
add_library(fw_cuda_lib STATIC ${CUDA_SOURCES})
target_link_libraries(fw_cuda_lib PUBLIC fw_cpu)
set_target_properties(fw_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# Executable: Benchmark Runner
# =============================================================================
add_executable(bench_runner src/bench_runner.cpp)
target_link_libraries(bench_runner PRIVATE fw_cuda_lib)
set_target_properties(bench_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin
)

# =============================================================================
# Executable: Tests Runner
# =============================================================================
add_executable(tests_runner
    src/tests_main.cpp
    src/tests.cpp
    src/tests_gpu.cu
    src/tests_runner.cu
)
target_link_libraries(tests_runner PRIVATE fw_cuda_lib)
set_target_properties(tests_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== fw_cuda Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "===================================")
message(STATUS "")
