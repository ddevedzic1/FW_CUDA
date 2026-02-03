#include "cuda_utils.cuh"
#include "fw_baseline_gpu.cuh"
#include "fw_tiling_gpu.cuh"
#include <stdexcept>

const std::map<std::string, AlgorithmFuncGPU> GPU_ALGORITHMS = {
    {"baseline_gpu", fwBaselineGPU},
    {"tiling_gpu", fwTilingGPU}
};

AlgorithmFuncGPU getGPUAlgorithmFunc(const std::string& algorithmName) {
    auto it = GPU_ALGORITHMS.find(algorithmName);
    if (it == GPU_ALGORITHMS.end()) {
        throw std::out_of_range("ERROR: Unsupported GPU algorithm: " + algorithmName);
    }
    return it->second;
}

bool isGPUAlgorithm(const std::string& algorithmName) {
    return GPU_ALGORITHMS.find(algorithmName) != GPU_ALGORITHMS.end();
}
