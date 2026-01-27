#include "tests_runner.h"
#include "tests.h"
#include "utils.h"
#include "cuda_utils.cuh"
#include "fw_baseline_cpu.h"
#include "timer.cuh"
#include <iostream>
#include <stdexcept>

namespace TestsRunner {

int executeTests(const std::string& algorithmName, int tileSize) {
    AlgorithmFuncCPU reference = fwBaselineCPU;

    if (isGPUAlgorithm(algorithmName)) {
        AlgorithmFuncGPU algorithm = nullptr;
        try {
            algorithm = getGPUAlgorithmFunc(algorithmName);
        } catch (const std::out_of_range& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }

        std::cout << "Warming up GPU...\n";
        Timer::warmupGPU();

        runTestsGPU(algorithm, reference, algorithmName, tileSize);
    } else {
        AlgorithmFuncCPU algorithm = nullptr;
        try {
            algorithm = getCPUAlgorithmFunc(algorithmName);
        } catch (const std::out_of_range& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }

        runTestsCPU(algorithm, reference, algorithmName, tileSize);
    }

    return 0;
}

} // namespace TestsRunner
