#include "utils.h"
#include "fw_baseline_cpu.h"
#include <stdexcept>

bool hasNegativeCycle(const WeightType* distMatrix, int n) {
    for (int i = 0; i < n; ++i) {
        if (distMatrix[i * n + i] < 0) {
            return true;
        }
    }
    return false;
}

const std::map<std::string, AlgorithmFuncCPU> CPU_ALGORITHMS = {
    {"baseline_cpu", fwBaselineCPU}
};

AlgorithmFuncCPU getCPUAlgorithmFunc(const std::string& algorithmName) {
    auto it = CPU_ALGORITHMS.find(algorithmName);
    if (it == CPU_ALGORITHMS.end()) {
        throw std::out_of_range("ERROR: Unsupported CPU algorithm: " + algorithmName);
    }
    return it->second;
}
