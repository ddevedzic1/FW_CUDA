#include "timer.cuh"
#include "cuda_utils.cuh"
#include <chrono>

namespace Timer {

__global__ void warmupKernel() {
}

void warmupGPU() {
    warmupKernel<<<1, 1>>>();
    checkKernelErrors();
    checkCudaErrors(cudaDeviceSynchronize());
}

double measureCPU(
    AlgorithmFuncCPU func,
    WeightType* graph,
    int n,
    int tileSize,
    int warmup_runs
) {
    for (int i = 0; i < warmup_runs; ++i) {
        func(graph, n, tileSize);
    }

    auto start = std::chrono::steady_clock::now();
    func(graph, n, tileSize);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double measureGPU(
    AlgorithmFuncGPU func,
    WeightType* d_graph,
    int n,
    int tileSize,
    int warmup_runs
) {
    for (int i = 0; i < warmup_runs; ++i) {
        func(d_graph, n, tileSize);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    func(d_graph, n, tileSize);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}

} // namespace Timer
