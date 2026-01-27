#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <map>
#include "algo_types.h"

// Error checking macro for CUDA API calls
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << "\n";
        std::cerr << "Error code: " << result << " (" << cudaGetErrorName(result) << ")\n";
        std::cerr << "Function: " << func << "\n";
        std::cerr << "Message: " << cudaGetErrorString(result) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// Error checking macro for kernel launches
#define checkKernelErrors() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "Kernel launch error at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::cerr << "Error: " << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU algorithm registry
extern const std::map<std::string, AlgorithmFuncGPU> GPU_ALGORITHMS;

// Returns GPU algorithm function by name, throws std::out_of_range if not found
AlgorithmFuncGPU getGPUAlgorithmFunc(const std::string& algorithmName);

// Returns true if the algorithm name refers to a GPU algorithm
bool isGPUAlgorithm(const std::string& algorithmName);
