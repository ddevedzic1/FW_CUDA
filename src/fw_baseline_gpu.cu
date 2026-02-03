#include "fw_baseline_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int BLOCK_SIZE = 16;

__global__ void fwBaselineKernel(WeightType* __restrict__ D, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int ij = i * n + j;
        D[ij] = min(D[ij], D[i * n + k] + D[k * n + j]);
    }
}

void fwBaselineGPU(WeightType* d_D, int n, int tileSize) {
    (void)tileSize;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    for (int k = 0; k < n; ++k) {
        fwBaselineKernel<<<gridDim, blockDim>>>(d_D, n, k);
        checkKernelErrors();
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
