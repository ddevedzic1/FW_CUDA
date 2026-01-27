#include "gpu_memory.cuh"
#include "cuda_utils.cuh"

namespace GPUMemory {

WeightType* allocate(int n) {
    WeightType* d_graph = nullptr;
    size_t size = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(WeightType);
    checkCudaErrors(cudaMalloc(&d_graph, size));
    return d_graph;
}

void copyToDevice(WeightType* d_graph, const WeightType* h_graph, int n) {
    size_t size = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(WeightType);
    checkCudaErrors(cudaMemcpy(d_graph, h_graph, size, cudaMemcpyHostToDevice));
}

void copyToHost(WeightType* h_graph, const WeightType* d_graph, int n) {
    size_t size = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(WeightType);
    checkCudaErrors(cudaMemcpy(h_graph, d_graph, size, cudaMemcpyDeviceToHost));
}

void free(WeightType* d_graph) {
    if (d_graph != nullptr) {
        checkCudaErrors(cudaFree(d_graph));
    }
}

} // namespace GPUMemory
