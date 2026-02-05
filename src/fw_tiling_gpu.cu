#include "fw_tiling_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int DEFAULT_TILE_SIZE = 32;

extern __shared__ WeightType sharedMem[];

template<int TILE>
__global__ void fwTilingDiagonalKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;

    int base = tile * TILE;
    int gI0 = base + r0;
    int gI1 = base + r1;
    int gJ0 = base + c0;
    int gJ1 = base + c1;

    WeightType* sharedD = sharedMem;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    sharedD[r0 * TILE + c0] = c00;
    sharedD[r0 * TILE + c1] = c01;
    sharedD[r1 * TILE + c0] = c10;
    sharedD[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType row0 = sharedD[r0 * TILE + k];
        WeightType row1 = sharedD[r1 * TILE + k];
        WeightType col0 = sharedD[k * TILE + c0];
        WeightType col1 = sharedD[k * TILE + c1];
        c00 = min(c00, row0 + col0);
        c01 = min(c01, row0 + col1);
        c10 = min(c10, row1 + col0);
        c11 = min(c11, row1 + col1);
        sharedD[r0 * TILE + c0] = c00;
        sharedD[r0 * TILE + c1] = c01;
        sharedD[r1 * TILE + c0] = c10;
        sharedD[r1 * TILE + c1] = c11;
        __syncthreads();
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

template<int TILE>
__global__ void fwTilingRowKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int diagBase = tile * TILE;
    int colTile = bx < tile ? bx : bx + 1;
    int colBase = colTile * TILE;

    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = colBase  + c0;
    int gJ1 = colBase  + c1;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + TILE * TILE;

    sharedDiag[r0 * TILE + c0] = (gI0 < n && diagBase + c0 < n) ? __ldg(&D[gI0 * n + diagBase + c0]) : INF;
    sharedDiag[r0 * TILE + c1] = (gI0 < n && diagBase + c1 < n) ? __ldg(&D[gI0 * n + diagBase + c1]) : INF;
    sharedDiag[r1 * TILE + c0] = (gI1 < n && diagBase + c0 < n) ? __ldg(&D[gI1 * n + diagBase + c0]) : INF;
    sharedDiag[r1 * TILE + c1] = (gI1 < n && diagBase + c1 < n) ? __ldg(&D[gI1 * n + diagBase + c1]) : INF;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    sharedCurr[r0 * TILE + c0] = c00;
    sharedCurr[r0 * TILE + c1] = c01;
    sharedCurr[r1 * TILE + c0] = c10;
    sharedCurr[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType diag0 = sharedDiag[r0 * TILE + k];
        WeightType diag1 = sharedDiag[r1 * TILE + k];
        WeightType cur0  = sharedCurr[k * TILE + c0];
        WeightType cur1  = sharedCurr[k * TILE + c1];
        c00 = min(c00, diag0 + cur0);
        c01 = min(c01, diag0 + cur1);
        c10 = min(c10, diag1 + cur0);
        c11 = min(c11, diag1 + cur1);
        sharedCurr[r0 * TILE + c0] = c00;
        sharedCurr[r0 * TILE + c1] = c01;
        sharedCurr[r1 * TILE + c0] = c10;
        sharedCurr[r1 * TILE + c1] = c11;
        __syncthreads();
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

template<int TILE>
__global__ void fwTilingColumnKernel(
    WeightType* __restrict__ D,
    int n,
    int tile
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int diagBase = tile * TILE;
    int rowTile = bx < tile ? bx : bx + 1;
    int rowBase = rowTile * TILE;

    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowBase  + r0;
    int gI1 = rowBase  + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedCurr = sharedMem + TILE * TILE;

    sharedDiag[r0 * TILE + c0] = (diagBase + r0 < n && gJ0 < n) ? __ldg(&D[(diagBase + r0) * n + gJ0]) : INF;
    sharedDiag[r0 * TILE + c1] = (diagBase + r0 < n && gJ1 < n) ? __ldg(&D[(diagBase + r0) * n + gJ1]) : INF;
    sharedDiag[r1 * TILE + c0] = (diagBase + r1 < n && gJ0 < n) ? __ldg(&D[(diagBase + r1) * n + gJ0]) : INF;
    sharedDiag[r1 * TILE + c1] = (diagBase + r1 < n && gJ1 < n) ? __ldg(&D[(diagBase + r1) * n + gJ1]) : INF;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    sharedCurr[r0 * TILE + c0] = c00;
    sharedCurr[r0 * TILE + c1] = c01;
    sharedCurr[r1 * TILE + c0] = c10;
    sharedCurr[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType cur0  = sharedCurr[r0 * TILE + k];
        WeightType cur1  = sharedCurr[r1 * TILE + k];
        WeightType diag0 = sharedDiag[k * TILE + c0];
        WeightType diag1 = sharedDiag[k * TILE + c1];
        c00 = min(c00, cur0 + diag0);
        c01 = min(c01, cur0 + diag1);
        c10 = min(c10, cur1 + diag0);
        c11 = min(c11, cur1 + diag1);
        sharedCurr[r0 * TILE + c0] = c00;
        sharedCurr[r0 * TILE + c1] = c01;
        sharedCurr[r1 * TILE + c0] = c10;
        sharedCurr[r1 * TILE + c1] = c11;
        __syncthreads();
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
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

    int diagBase = tile * TILE;

    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowTile * TILE + r0;
    int gI1 = rowTile * TILE + r1;
    int gJ0 = colTile * TILE + c0;
    int gJ1 = colTile * TILE + c1;

    WeightType* sharedRow = sharedMem;
    WeightType* sharedCol = sharedMem + TILE * TILE;

    sharedRow[r0 * TILE + c0] = (gI0 < n && diagBase + c0 < n) ? __ldg(&D[gI0 * n + diagBase + c0]) : INF;
    sharedRow[r0 * TILE + c1] = (gI0 < n && diagBase + c1 < n) ? __ldg(&D[gI0 * n + diagBase + c1]) : INF;
    sharedRow[r1 * TILE + c0] = (gI1 < n && diagBase + c0 < n) ? __ldg(&D[gI1 * n + diagBase + c0]) : INF;
    sharedRow[r1 * TILE + c1] = (gI1 < n && diagBase + c1 < n) ? __ldg(&D[gI1 * n + diagBase + c1]) : INF;

    sharedCol[r0 * TILE + c0] = (diagBase + r0 < n && gJ0 < n) ? __ldg(&D[(diagBase + r0) * n + gJ0]) : INF;
    sharedCol[r0 * TILE + c1] = (diagBase + r0 < n && gJ1 < n) ? __ldg(&D[(diagBase + r0) * n + gJ1]) : INF;
    sharedCol[r1 * TILE + c0] = (diagBase + r1 < n && gJ0 < n) ? __ldg(&D[(diagBase + r1) * n + gJ0]) : INF;
    sharedCol[r1 * TILE + c1] = (diagBase + r1 < n && gJ1 < n) ? __ldg(&D[(diagBase + r1) * n + gJ1]) : INF;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        WeightType row0 = sharedRow[r0 * TILE + k];
        WeightType row1 = sharedRow[r1 * TILE + k];
        WeightType col0 = sharedCol[k * TILE + c0];
        WeightType col1 = sharedCol[k * TILE + c1];
        c00 = min(c00, row0 + col0);
        c01 = min(c01, row0 + col1);
        c10 = min(c10, row1 + col0);
        c11 = min(c11, row1 + col1);
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

void fwTilingGPU(WeightType* d_D, int n, int tileSize, int kappa) {
    (void)tileSize;
    (void)kappa;

    constexpr int TILE = DEFAULT_TILE_SIZE;

    int numTiles = (n + TILE - 1) / TILE;
    size_t tileBytes = static_cast<size_t>(TILE) * TILE * sizeof(WeightType);

    dim3 blockDim(TILE / 2, TILE / 2);

    for (int tile = 0; tile < numTiles; ++tile) {
        fwTilingDiagonalKernel<TILE><<<1, blockDim, tileBytes>>>(d_D, n, tile);
        checkKernelErrors();

        if (numTiles > 1) {
            dim3 gridRow(numTiles - 1);
            fwTilingRowKernel<TILE><<<gridRow, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();

            dim3 gridCol(numTiles - 1);
            fwTilingColumnKernel<TILE><<<gridCol, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();

            dim3 gridOthers(numTiles - 1, numTiles - 1);
            fwTilingOthersKernel<TILE><<<gridOthers, blockDim, 2 * tileBytes>>>(d_D, n, tile);
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
