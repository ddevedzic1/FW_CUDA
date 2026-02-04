#include "fw_multi_layer_tiling_optimized_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int DEFAULT_TILE_SIZE = 32;
constexpr int DEFAULT_KAPPA     = 4;
constexpr int MAX_KAPPA         = 8;

extern __shared__ WeightType sharedMem[];

template<int TILE>
__global__ void fwMLOptLeadBlockKernel(
    WeightType* __restrict__ D,
    int n,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int currentK  = kBlockBase + l;
    int diagBase  = currentK * TILE;
    int globalI   = diagBase + ty;
    int globalJ   = diagBase + tx;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCol  = sharedMem + TILE * TILE;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    for (int m = 0; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    sharedDiag[ty * TILE + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        sharedDiag[ty * TILE + tx] = min(
            sharedDiag[ty * TILE + tx],
            sharedDiag[ty * TILE + k] + sharedDiag[k * TILE + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedDiag[ty * TILE + tx];
    }
}

template<int TILE>
__global__ void fwMLOptLeadRowKernel(
    WeightType* __restrict__ D,
    int n,
    int numTiles,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;
    int colTile  = (bx < currentK) ? bx : bx + 1;

    int diagBase = currentK * TILE;
    int colBase  = colTile  * TILE;
    int globalI  = diagBase + ty;
    int globalJ  = colBase  + tx;

    WeightType* sharedDiag  = sharedMem;
    WeightType* sharedCurr  = sharedMem +     TILE * TILE;
    WeightType* sharedDDCol = sharedMem + 2 * TILE * TILE;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    int mStart = (colTile >= kBlockBase && colTile < currentK)
        ? (colTile - kBlockBase + 1) : 0;

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedDDCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedDDCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    sharedDiag[ty * TILE + tx] = (diagBase + ty < n && diagBase + tx < n)
        ? __ldg(&D[(diagBase + ty) * n + diagBase + tx]) : INF;
    sharedCurr[ty * TILE + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        sharedCurr[ty * TILE + tx] = min(
            sharedCurr[ty * TILE + tx],
            sharedDiag[ty * TILE + k] + sharedCurr[k * TILE + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedCurr[ty * TILE + tx];
    }
}

template<int TILE>
__global__ void fwMLOptLeadColumnKernel(
    WeightType* __restrict__ D,
    int n,
    int numTiles,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK = kBlockBase + l;
    int rowTile  = (bx < currentK) ? bx : bx + 1;

    int diagBase = currentK * TILE;
    int rowBase  = rowTile  * TILE;
    int globalI  = rowBase  + ty;
    int globalJ  = diagBase + tx;

    WeightType* sharedDiag  = sharedMem;
    WeightType* sharedCurr  = sharedMem +     TILE * TILE;
    WeightType* sharedDDCol = sharedMem + 2 * TILE * TILE;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    int mStart = (rowTile >= kBlockBase && rowTile < currentK)
        ? (rowTile - kBlockBase + 1) : 0;

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedDDCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedDDCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    sharedDiag[ty * TILE + tx] = (diagBase + ty < n && diagBase + tx < n)
        ? __ldg(&D[(diagBase + ty) * n + diagBase + tx]) : INF;
    sharedCurr[ty * TILE + tx] = currVal;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        sharedCurr[ty * TILE + tx] = min(
            sharedCurr[ty * TILE + tx],
            sharedCurr[ty * TILE + k] + sharedDiag[k * TILE + tx]
        );
        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = sharedCurr[ty * TILE + tx];
    }
}

template<int TILE>
__global__ void fwMLOptLeadRowReverseKernel(
    WeightType* __restrict__ D,
    int n,
    int numTiles,
    int kBlockBase,
    int blocksInStage,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK     = kBlockBase + l;
    int maxLeftTiles = currentK + 1;
    int rightStart   = kBlockBase + blocksInStage;

    int colTile = (bx < maxLeftTiles)
        ? bx
        : rightStart + (bx - maxLeftTiles);

    int diagBase = currentK * TILE;
    int colBase  = colTile  * TILE;
    int globalI  = diagBase + ty;
    int globalJ  = colBase  + tx;

    WeightType* sharedDDCol = sharedMem;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedDDCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedDDCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

template<int TILE>
__global__ void fwMLOptLeadColumnReverseKernel(
    WeightType* __restrict__ D,
    int n,
    int numTiles,
    int kBlockBase,
    int blocksInStage,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int currentK     = kBlockBase + l;
    int maxLeftTiles = currentK;
    int rightStart   = kBlockBase + blocksInStage;

    int rowTile = (bx < maxLeftTiles)
        ? bx
        : rightStart + (bx - maxLeftTiles);

    int diagBase = currentK * TILE;
    int rowBase  = rowTile  * TILE;
    int globalI  = rowBase  + ty;
    int globalJ  = diagBase + tx;

    WeightType* sharedDDCol = sharedMem;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedDDCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedDDCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

template<int TILE>
__global__ void fwMLOptRestBlocksKernel(
    WeightType* __restrict__ D,
    int n,
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

    int globalI = rowTile * TILE + ty;
    int globalJ = colTile * TILE + tx;

    WeightType* sharedCol = sharedMem;

    WeightType currVal = (globalI < n && globalJ < n) ? __ldg(&D[globalI * n + globalJ]) : INF;

    for (int m = 0; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        WeightType rowVal = (globalI < n && mBase + tx < n)
            ? __ldg(&D[globalI * n + mBase + tx]) : INF;
        sharedCol[ty * TILE + tx] = (mBase + ty < n && globalJ < n)
            ? __ldg(&D[(mBase + ty) * n + globalJ]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            currVal = min(currVal, __shfl_sync(0xFFFFFFFF, rowVal, k) + sharedCol[k * TILE + tx]);
        }

        __syncthreads();
    }

    if (globalI < n && globalJ < n) {
        D[globalI * n + globalJ] = currVal;
    }
}

void fwMultiLayerTilingOptimizedGPU(WeightType* d_D, int n, int tileSize, int kappa) {
    if (tileSize <= 0 || tileSize > 32) {
        tileSize = DEFAULT_TILE_SIZE;
    }
    if (kappa <= 0 || kappa > MAX_KAPPA) {
        kappa = DEFAULT_KAPPA;
    }

    constexpr int TILE = 32;

    int numTiles = (n + TILE - 1) / TILE;

    dim3 blockDim(TILE, TILE);
    size_t tileBytes = static_cast<size_t>(TILE) * TILE * sizeof(WeightType);

    for (int kBlockBase = 0; kBlockBase < numTiles; kBlockBase += kappa) {
        int blocksInStage = min(kappa, numTiles - kBlockBase);

        for (int l = 0; l < blocksInStage; ++l) {
            fwMLOptLeadBlockKernel<TILE><<<1, blockDim, 2 * tileBytes>>>(
                d_D, n, kBlockBase, l
            );
            checkKernelErrors();

            int numRowColTiles = numTiles - 1;
            if (numRowColTiles > 0) {
                fwMLOptLeadRowKernel<TILE><<<numRowColTiles, blockDim, 3 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();

                fwMLOptLeadColumnKernel<TILE><<<numRowColTiles, blockDim, 3 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();
            }
        }

        for (int l = blocksInStage - 2; l >= 0; --l) {
            int currentK = kBlockBase + l;

            int numRowTiles = (currentK + 1) + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numRowTiles > 0) {
                fwMLOptLeadRowReverseKernel<TILE><<<numRowTiles, blockDim, tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }

            int numColTiles = currentK + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numColTiles > 0) {
                fwMLOptLeadColumnReverseKernel<TILE><<<numColTiles, blockDim, tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }
        }

        if (blocksInStage < numTiles) {
            int numActiveTiles = numTiles - blocksInStage;
            dim3 gridRest(numActiveTiles, numActiveTiles);

            fwMLOptRestBlocksKernel<TILE><<<gridRest, blockDim, tileBytes>>>(
                d_D, n, numTiles, kBlockBase, blocksInStage
            );
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
