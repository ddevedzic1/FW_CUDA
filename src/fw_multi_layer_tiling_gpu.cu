#include "fw_multi_layer_tiling_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int DEFAULT_TILE_SIZE = 32;
constexpr int DEFAULT_KAPPA = 4;
constexpr int MAX_KAPPA = 8;

extern __shared__ WeightType sharedMem[];

__global__ void fwMultiLayerLeadBlockKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int currentK = kBlockBase + l;
    int diagBase = currentK * tileSize;
    int globalI = diagBase + ty;
    int globalJ = diagBase + tx;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedRow = sharedMem + tileSize * tileSize;
    WeightType* sharedCol = sharedMem + 2 * tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    for (int m = 0; m < l; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedRow[ty * tileSize + k] + sharedCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    sharedDiag[ty * tileSize + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedDiag[ty * tileSize + tx] = min(
            sharedDiag[ty * tileSize + tx],
            sharedDiag[ty * tileSize + k] + sharedDiag[k * tileSize + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedDiag[ty * tileSize + tx];
    }
}

__global__ void fwMultiLayerLeadRowKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int numTiles,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;

    int colTile = (bx < currentK) ? bx : bx + 1;

    int diagBase = currentK * tileSize;
    int colBase = colTile * tileSize;
    int globalI = diagBase + ty;
    int globalJ = colBase + tx;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + tileSize * tileSize;
    WeightType* sharedDDRow = sharedMem + 2 * tileSize * tileSize;
    WeightType* sharedDDCol = sharedMem + 3 * tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    int mStart;
    if (colTile >= kBlockBase && colTile < currentK) {
        mStart = colTile - kBlockBase + 1;
    } else {
        mStart = 0;
    }

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedDDRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedDDCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedDDRow[ty * tileSize + k] + sharedDDCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    int diagGlobalI = diagBase + ty;
    int diagGlobalJ = diagBase + tx;
    sharedDiag[ty * tileSize + tx] = (diagGlobalI < n && diagGlobalJ < n)
        ? D[diagGlobalI * n + diagGlobalJ]
        : INF;

    sharedCurr[ty * tileSize + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedCurr[ty * tileSize + tx] = min(
            sharedCurr[ty * tileSize + tx],
            sharedDiag[ty * tileSize + k] + sharedCurr[k * tileSize + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedCurr[ty * tileSize + tx];
    }
}

__global__ void fwMultiLayerLeadColumnKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int numTiles,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;

    int rowTile = (bx < currentK) ? bx : bx + 1;

    int diagBase = currentK * tileSize;
    int rowBase = rowTile * tileSize;
    int globalI = rowBase + ty;
    int globalJ = diagBase + tx;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + tileSize * tileSize;
    WeightType* sharedDDRow = sharedMem + 2 * tileSize * tileSize;
    WeightType* sharedDDCol = sharedMem + 3 * tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    int mStart;
    if (rowTile >= kBlockBase && rowTile < currentK) {
        mStart = rowTile - kBlockBase + 1;
    } else {
        mStart = 0;
    }

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedDDRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedDDCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedDDRow[ty * tileSize + k] + sharedDDCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    int diagGlobalI = diagBase + ty;
    int diagGlobalJ = diagBase + tx;
    sharedDiag[ty * tileSize + tx] = (diagGlobalI < n && diagGlobalJ < n)
        ? D[diagGlobalI * n + diagGlobalJ]
        : INF;

    sharedCurr[ty * tileSize + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        sharedCurr[ty * tileSize + tx] = min(
            sharedCurr[ty * tileSize + tx],
            sharedCurr[ty * tileSize + k] + sharedDiag[k * tileSize + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedCurr[ty * tileSize + tx];
    }
}

__global__ void fwMultiLayerLeadRowReverseKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int numTiles,
    int kBlockBase,
    int blocksInStage,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;

    int maxLeftTiles = currentK + 1;
    int rightStart = kBlockBase + blocksInStage;

    int colTile;
    if (bx < maxLeftTiles) {
        colTile = bx;
    } else {
        colTile = rightStart + (bx - maxLeftTiles);
    }

    int diagBase = currentK * tileSize;
    int colBase = colTile * tileSize;
    int globalI = diagBase + ty;
    int globalJ = colBase + tx;

    WeightType* sharedDDRow = sharedMem;
    WeightType* sharedDDCol = sharedMem + tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedDDRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedDDCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedDDRow[ty * tileSize + k] + sharedDDCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

__global__ void fwMultiLayerLeadColumnReverseKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int numTiles,
    int kBlockBase,
    int blocksInStage,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;

    int maxLeftTiles = currentK;
    int rightStart = kBlockBase + blocksInStage;

    int rowTile;
    if (bx < maxLeftTiles) {
        rowTile = bx;
    } else {
        rowTile = rightStart + (bx - maxLeftTiles);
    }

    int diagBase = currentK * tileSize;
    int rowBase = rowTile * tileSize;
    int globalI = rowBase + ty;
    int globalJ = diagBase + tx;

    WeightType* sharedDDRow = sharedMem;
    WeightType* sharedDDCol = sharedMem + tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedDDRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedDDCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedDDRow[ty * tileSize + k] + sharedDDCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

__global__ void fwMultiLayerRestBlocksKernel(
    WeightType* __restrict__ D,
    int n,
    int tileSize,
    int numTiles,
    int kBlockBase,
    int blocksInStage
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int numLeftTiles = kBlockBase;

    int colTile = (bx < numLeftTiles) ? bx : (kBlockBase + blocksInStage + (bx - numLeftTiles));
    int rowTile = (by < numLeftTiles) ? by : (kBlockBase + blocksInStage + (by - numLeftTiles));

    int rowBase = rowTile * tileSize;
    int colBase = colTile * tileSize;
    int globalI = rowBase + ty;
    int globalJ = colBase + tx;

    WeightType* sharedRow = sharedMem;
    WeightType* sharedCol = sharedMem + tileSize * tileSize;

    WeightType currVal = (globalI < n && globalJ < n) ? D[globalI * n + globalJ] : INF;

    for (int m = 0; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * tileSize;

        int rowGlobalK = mBase + tx;
        sharedRow[ty * tileSize + tx] = (globalI < n && rowGlobalK < n)
            ? D[globalI * n + rowGlobalK]
            : INF;

        int colGlobalK = mBase + ty;
        sharedCol[ty * tileSize + tx] = (colGlobalK < n && globalJ < n)
            ? D[colGlobalK * n + globalJ]
            : INF;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k) {
            currVal = min(currVal, sharedRow[ty * tileSize + k] + sharedCol[k * tileSize + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

void fwMultiLayerTilingGPU(WeightType* d_D, int n, int tileSize, int kappa) {
    if (tileSize <= 0 || tileSize > 32) {
        tileSize = DEFAULT_TILE_SIZE;
    }
    if (kappa <= 0 || kappa > MAX_KAPPA) {
        kappa = DEFAULT_KAPPA;
    }

    int numTiles = (n + tileSize - 1) / tileSize;

    dim3 blockDim(tileSize, tileSize);
    size_t tileBytes = static_cast<size_t>(tileSize) * tileSize * sizeof(WeightType);

    for (int kBlockBase = 0; kBlockBase < numTiles; kBlockBase += kappa) {
        int blocksInStage = min(kappa, numTiles - kBlockBase);

        for (int l = 0; l < blocksInStage; ++l) {
            fwMultiLayerLeadBlockKernel<<<1, blockDim, 3 * tileBytes>>>(
                d_D, n, tileSize, kBlockBase, l
            );
            checkKernelErrors();

            int numRowColTiles = numTiles - 1;
            if (numRowColTiles > 0) {
                fwMultiLayerLeadRowKernel<<<numRowColTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, tileSize, numTiles, kBlockBase, l
                );
                checkKernelErrors();

                fwMultiLayerLeadColumnKernel<<<numRowColTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, tileSize, numTiles, kBlockBase, l
                );
                checkKernelErrors();
            }
        }

        for (int l = blocksInStage - 2; l >= 0; --l) {
            int currentK = kBlockBase + l;

            int numRowTiles = (currentK + 1) + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numRowTiles > 0) {
                fwMultiLayerLeadRowReverseKernel<<<numRowTiles, blockDim, 2 * tileBytes>>>(
                    d_D, n, tileSize, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }

            int numColTiles = currentK + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numColTiles > 0) {
                fwMultiLayerLeadColumnReverseKernel<<<numColTiles, blockDim, 2 * tileBytes>>>(
                    d_D, n, tileSize, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }
        }

        if (blocksInStage < numTiles) {
            int numActiveTiles = numTiles - blocksInStage;
            dim3 gridRest(numActiveTiles, numActiveTiles);

            fwMultiLayerRestBlocksKernel<<<gridRest, blockDim, 2 * tileBytes>>>(
                d_D, n, tileSize, numTiles, kBlockBase, blocksInStage
            );
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}

