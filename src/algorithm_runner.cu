#include "algorithm_runner.h"
#include "utils.h"
#include "graph_generator.h"
#include "gpu_memory.cuh"
#include "timer.cuh"
#include "cuda_utils.cuh"
#include <iostream>
#include <iomanip>

namespace AlgorithmRunner {

int executeBenchmark(
    const std::string& algorithmName,
    int n,
    int tileSize,
    double density,
    unsigned int seed
) {
    if (n <= 0 || n > MAX_N) {
        std::cerr << "ERROR: Graph size (n) must be between 1 and " << MAX_N << ". Aborting.\n";
        return 1;
    }

    bool isGPU = isGPUAlgorithm(algorithmName);

    std::cout << "\n========================================\n";
    std::cout << "BENCHMARK: " << algorithmName << "\n";
    std::cout << "========================================\n";
    std::cout << "Graph size: " << n << " vertices\n";
    std::cout << "Edge density: " << std::fixed << std::setprecision(1) << (density * 100.0) << "%\n";
    std::cout << "Tile size: " << tileSize << "\n";
    std::cout << "Algorithm type: " << (isGPU ? "GPU" : "CPU") << "\n";
    if (seed == 0) {
        std::cout << "Seed: Random (Non-reproducible)\n";
    } else {
        std::cout << "Seed: " << seed << " (Reproducible)\n";
    }
    std::cout << "\n";

    std::cout << "Generating graph...\n";
    WeightType* h_graph = generateAlignedGraph1D(n, density, seed);
    if (h_graph == nullptr) {
        std::cerr << "ERROR: Failed to generate graph.\n";
        return 1;
    }

    double elapsed = 0.0;
    WeightType* d_graph = nullptr;

    try {
        if (isGPU) {
            AlgorithmFuncGPU gpuFunc = getGPUAlgorithmFunc(algorithmName);

            std::cout << "Warming up GPU...\n";
            Timer::warmupGPU();

            std::cout << "Copying graph to GPU...\n";
            d_graph = GPUMemory::allocate(n);
            GPUMemory::copyToDevice(d_graph, h_graph, n);

            std::cout << "Running " << algorithmName << "...\n";
            elapsed = Timer::measureGPU(gpuFunc, d_graph, n, tileSize, 0);

            GPUMemory::copyToHost(h_graph, d_graph, n);
            GPUMemory::free(d_graph);
            d_graph = nullptr;

        } else {
            AlgorithmFuncCPU cpuFunc = getCPUAlgorithmFunc(algorithmName);

            std::cout << "Running " << algorithmName << "...\n";
            elapsed = Timer::measureCPU(cpuFunc, h_graph, n, tileSize, 0);
        }
    } catch (const std::exception& e) {
        if (d_graph != nullptr) {
            GPUMemory::free(d_graph);
        }
        ALIGNED_FREE(h_graph);
        throw;
    }

    std::cout << "\nResults:\n";
    std::cout << "--------\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsed << " seconds\n";

    ALIGNED_FREE(h_graph);

    return 0;
}

} // namespace AlgorithmRunner
