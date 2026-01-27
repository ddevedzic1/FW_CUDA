#include "fw_baseline_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int BLOCK_DIM = 16;

__global__ void fwBaselineKernel(WeightType* D, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int ij = i * n + j;
        int ik = i * n + k;
        int kj = k * n + j;

        WeightType d_ik = D[ik];
        WeightType d_kj = D[kj];

        if (d_ik != INF && d_kj != INF) {
            WeightType newDist = d_ik + d_kj;
            if (newDist < D[ij]) {
                D[ij] = newDist;
            }
        }
    }
}

void fwBaselineGPU(WeightType* d_D, int n, int tileSize) {
    (void)tileSize;

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks(
        (n + BLOCK_DIM - 1) / BLOCK_DIM,
        (n + BLOCK_DIM - 1) / BLOCK_DIM
    );

    for (int k = 0; k < n; ++k) {
        fwBaselineKernel<<<numBlocks, threadsPerBlock>>>(d_D, n, k);
        checkKernelErrors();
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
