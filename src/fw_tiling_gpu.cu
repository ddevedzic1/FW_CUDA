#include "fw_tiling_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int DEFAULT_TILE_SIZE = 32;

extern __shared__ WeightType sharedMem[];

__global__ void fwTilingDiagonalKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tileSize = blockDim.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    WeightType* sharedD = sharedMem;

    int base = tile * tileSize;
    int i = base + ty;
    int j = base + tx;

    sharedD[ty * tileSize + tx] = (i < n && j < n) ? __ldg(&D[i * n + j]) : INF;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedD[ty * tileSize + tx] = min(sharedD[ty * tileSize + tx],
                                          sharedD[ty * tileSize + k] + sharedD[k * tileSize + tx]);
        __syncthreads();
    }

    if (i < n && j < n) {
        D[i * n + j] = sharedD[ty * tileSize + tx];
    }
}

__global__ void fwTilingRowKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tileSize = blockDim.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + tileSize * tileSize;

    int diagBase = tile * tileSize;
    int colTile = bx < tile ? bx : bx + 1;

    int i = diagBase + ty;
    int j = colTile * tileSize + tx;

    sharedDiag[ty * tileSize + tx] = (i < n && diagBase + tx < n) ? __ldg(&D[i * n + diagBase + tx]) : INF;
    sharedCurr[ty * tileSize + tx] = (i < n && j < n) ? __ldg(&D[i * n + j]) : INF;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedCurr[ty * tileSize + tx] = min(sharedCurr[ty * tileSize + tx],
                                             sharedDiag[ty * tileSize + k] + sharedCurr[k * tileSize + tx]);
        __syncthreads();
    }

    if (i < n && j < n) {
        D[i * n + j] = sharedCurr[ty * tileSize + tx];
    }
}

__global__ void fwTilingColumnKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tileSize = blockDim.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + tileSize * tileSize;

    int diagBase = tile * tileSize;
    int rowTile = bx < tile ? bx : bx + 1;

    int i = rowTile * tileSize + ty;
    int j = diagBase + tx;

    sharedDiag[ty * tileSize + tx] = (diagBase + ty < n && j < n) ? __ldg(&D[(diagBase + ty) * n + j]) : INF;
    sharedCurr[ty * tileSize + tx] = (i < n && j < n) ? __ldg(&D[i * n + j]) : INF;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedCurr[ty * tileSize + tx] = min(sharedCurr[ty * tileSize + tx],
                                             sharedCurr[ty * tileSize + k] + sharedDiag[k * tileSize + tx]);
        __syncthreads();
    }

    if (i < n && j < n) {
        D[i * n + j] = sharedCurr[ty * tileSize + tx];
    }
}

template<int TILE>
__global__ void fwTilingOthersKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int colTile = bx < tile ? bx : bx + 1;
    int rowTile = by < tile ? by : by + 1;

    WeightType* sharedRow = sharedMem;
    WeightType* sharedCol = sharedMem + TILE * TILE;

    int diagBase = tile * TILE;

    int currI = rowTile * TILE + ty;
    int currJ = colTile * TILE + tx;

    sharedRow[ty * TILE + tx] = (currI < n && diagBase + tx < n) ? __ldg(&D[currI * n + diagBase + tx]) : INF;
    sharedCol[ty * TILE + tx] = (diagBase + ty < n && currJ < n) ? __ldg(&D[(diagBase + ty) * n + currJ]) : INF;

    WeightType currVal = (currI < n && currJ < n) ? __ldg(&D[currI * n + currJ]) : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        currVal = min(currVal, sharedRow[ty * TILE + k] + sharedCol[k * TILE + tx]);
    }

    if (currI < n && currJ < n) {
        D[currI * n + currJ] = currVal;
    }
}

void fwTilingGPU(WeightType* d_D, int n, int tileSize, int kappa) {
    (void)kappa;

    if (tileSize <= 0 || tileSize > 32) {
        tileSize = DEFAULT_TILE_SIZE;
    }

    int numTiles = (n + tileSize - 1) / tileSize;
    size_t tileBytes = static_cast<size_t>(tileSize) * tileSize * sizeof(WeightType);

    dim3 blockDim(tileSize, tileSize);

    for (int tile = 0; tile < numTiles; ++tile) {
        fwTilingDiagonalKernel<<<1, blockDim, tileBytes>>>(d_D, n, tile);
        checkKernelErrors();

        if (numTiles > 1) {
            dim3 gridRow(numTiles - 1);
            fwTilingRowKernel<<<gridRow, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();

            dim3 gridCol(numTiles - 1);
            fwTilingColumnKernel<<<gridCol, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();

            dim3 gridOthers(numTiles - 1, numTiles - 1);
            fwTilingOthersKernel<DEFAULT_TILE_SIZE><<<gridOthers, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
