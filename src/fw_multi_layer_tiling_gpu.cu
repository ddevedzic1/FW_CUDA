#include "fw_multi_layer_tiling_gpu.cuh"
#include "cuda_utils.cuh"

constexpr int DEFAULT_TILE_SIZE = 32;
constexpr int DEFAULT_KAPPA = 4;
constexpr int MAX_KAPPA = 16;

extern __shared__ WeightType sharedMem[];

template<int TILE>
__global__ void fwMultiLayerLeadBlockKernel(
    WeightType* __restrict__ D,
    int n,
    int kBlockBase,
    int l
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int currentK  = kBlockBase + l;
    int diagBase  = currentK * TILE;
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDiag = sharedMem;
    WeightType* sharedRow  = sharedMem +     TILE * TILE;
    WeightType* sharedCol  = sharedMem + 2 * TILE * TILE;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    for (int m = 0; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

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
    }

    sharedDiag[r0 * TILE + c0] = c00;
    sharedDiag[r0 * TILE + c1] = c01;
    sharedDiag[r1 * TILE + c0] = c10;
    sharedDiag[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType row0 = sharedDiag[r0 * TILE + k];
        WeightType row1 = sharedDiag[r1 * TILE + k];
        WeightType col0 = sharedDiag[k * TILE + c0];
        WeightType col1 = sharedDiag[k * TILE + c1];
        c00 = min(c00, row0 + col0);
        c01 = min(c01, row0 + col1);
        c10 = min(c10, row1 + col0);
        c11 = min(c11, row1 + col1);
        sharedDiag[r0 * TILE + c0] = c00;
        sharedDiag[r0 * TILE + c1] = c01;
        sharedDiag[r1 * TILE + c0] = c10;
        sharedDiag[r1 * TILE + c1] = c11;
        __syncthreads();
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

template<int TILE>
__global__ void fwMultiLayerLeadRowKernel(
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = colBase  + c0;
    int gJ1 = colBase  + c1;

    WeightType* sharedDiag  = sharedMem;
    WeightType* sharedCurr  = sharedMem +     TILE * TILE;
    WeightType* sharedDDRow = sharedMem + 2 * TILE * TILE;
    WeightType* sharedDDCol = sharedMem + 3 * TILE * TILE;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int mStart = (colTile >= kBlockBase && colTile < currentK)
        ? (colTile - kBlockBase + 1) : 0;

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedDDRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedDDCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[r0 * TILE + k];
            WeightType row1 = sharedDDRow[r1 * TILE + k];
            WeightType col0 = sharedDDCol[k * TILE + c0];
            WeightType col1 = sharedDDCol[k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }
    }

    sharedDiag[r0 * TILE + c0] = (diagBase + r0 < n && diagBase + c0 < n) ? __ldg(&D[(diagBase + r0) * n + diagBase + c0]) : INF;
    sharedDiag[r0 * TILE + c1] = (diagBase + r0 < n && diagBase + c1 < n) ? __ldg(&D[(diagBase + r0) * n + diagBase + c1]) : INF;
    sharedDiag[r1 * TILE + c0] = (diagBase + r1 < n && diagBase + c0 < n) ? __ldg(&D[(diagBase + r1) * n + diagBase + c0]) : INF;
    sharedDiag[r1 * TILE + c1] = (diagBase + r1 < n && diagBase + c1 < n) ? __ldg(&D[(diagBase + r1) * n + diagBase + c1]) : INF;
    sharedCurr[r0 * TILE + c0] = c00;
    sharedCurr[r0 * TILE + c1] = c01;
    sharedCurr[r1 * TILE + c0] = c10;
    sharedCurr[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType diag0 = sharedDiag[r0 * TILE + k];
        WeightType diag1 = sharedDiag[r1 * TILE + k];
        c00 = min(c00, diag0 + sharedCurr[k * TILE + c0]);
        c01 = min(c01, diag0 + sharedCurr[k * TILE + c1]);
        c10 = min(c10, diag1 + sharedCurr[k * TILE + c0]);
        c11 = min(c11, diag1 + sharedCurr[k * TILE + c1]);
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
__global__ void fwMultiLayerLeadColumnKernel(
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowBase  + r0;
    int gI1 = rowBase  + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDiag  = sharedMem;
    WeightType* sharedCurr  = sharedMem +     TILE * TILE;
    WeightType* sharedDDRow = sharedMem + 2 * TILE * TILE;
    WeightType* sharedDDCol = sharedMem + 3 * TILE * TILE;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int mStart = (rowTile >= kBlockBase && rowTile < currentK)
        ? (rowTile - kBlockBase + 1) : 0;

    for (int m = mStart; m < l; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedDDRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedDDCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[r0 * TILE + k];
            WeightType row1 = sharedDDRow[r1 * TILE + k];
            WeightType col0 = sharedDDCol[k * TILE + c0];
            WeightType col1 = sharedDDCol[k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }
    }

    sharedDiag[r0 * TILE + c0] = (diagBase + r0 < n && diagBase + c0 < n) ? __ldg(&D[(diagBase + r0) * n + diagBase + c0]) : INF;
    sharedDiag[r0 * TILE + c1] = (diagBase + r0 < n && diagBase + c1 < n) ? __ldg(&D[(diagBase + r0) * n + diagBase + c1]) : INF;
    sharedDiag[r1 * TILE + c0] = (diagBase + r1 < n && diagBase + c0 < n) ? __ldg(&D[(diagBase + r1) * n + diagBase + c0]) : INF;
    sharedDiag[r1 * TILE + c1] = (diagBase + r1 < n && diagBase + c1 < n) ? __ldg(&D[(diagBase + r1) * n + diagBase + c1]) : INF;
    sharedCurr[r0 * TILE + c0] = c00;
    sharedCurr[r0 * TILE + c1] = c01;
    sharedCurr[r1 * TILE + c0] = c10;
    sharedCurr[r1 * TILE + c1] = c11;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        WeightType diag0 = sharedDiag[k * TILE + c0];
        WeightType diag1 = sharedDiag[k * TILE + c1];
        c00 = min(c00, sharedCurr[r0 * TILE + k] + diag0);
        c01 = min(c01, sharedCurr[r0 * TILE + k] + diag1);
        c10 = min(c10, sharedCurr[r1 * TILE + k] + diag0);
        c11 = min(c11, sharedCurr[r1 * TILE + k] + diag1);
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
__global__ void fwMultiLayerLeadRowReverseKernel(
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = colBase  + c0;
    int gJ1 = colBase  + c1;

    WeightType* sharedDDRow = sharedMem;
    WeightType* sharedDDCol = sharedMem + TILE * TILE;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedDDRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedDDCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[r0 * TILE + k];
            WeightType row1 = sharedDDRow[r1 * TILE + k];
            WeightType col0 = sharedDDCol[k * TILE + c0];
            WeightType col1 = sharedDDCol[k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

template<int TILE>
__global__ void fwMultiLayerLeadColumnReverseKernel(
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowBase  + r0;
    int gI1 = rowBase  + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDDRow = sharedMem;
    WeightType* sharedDDCol = sharedMem + TILE * TILE;

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    for (int m = l + 1; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedDDRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedDDCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[r0 * TILE + k];
            WeightType row1 = sharedDDRow[r1 * TILE + k];
            WeightType col0 = sharedDDCol[k * TILE + c0];
            WeightType col1 = sharedDDCol[k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

template<int TILE>
__global__ void fwMultiLayerRestBlocksKernel(
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

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    for (int m = 0; m < blocksInStage; ++m) {
        int mBase = (kBlockBase + m) * TILE;

        sharedRow[r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedRow[r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedRow[r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedRow[r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;

        sharedCol[r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedCol[r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedCol[r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedCol[r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;

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
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
}

void fwMultiLayerTilingGPU(WeightType* d_D, int n, int tileSize, int kappa) {
    if (tileSize <= 0 || tileSize > 32) {
        tileSize = DEFAULT_TILE_SIZE;
    }
    if (kappa <= 0 || kappa > MAX_KAPPA) {
        kappa = DEFAULT_KAPPA;
    }

    constexpr int TILE = 32;

    int numTiles = (n + TILE - 1) / TILE;

    dim3 blockDim(TILE / 2, TILE / 2);
    size_t tileBytes = static_cast<size_t>(TILE) * TILE * sizeof(WeightType);

    for (int kBlockBase = 0; kBlockBase < numTiles; kBlockBase += kappa) {
        int blocksInStage = min(kappa, numTiles - kBlockBase);

        for (int l = 0; l < blocksInStage; ++l) {
            fwMultiLayerLeadBlockKernel<TILE><<<1, blockDim, 3 * tileBytes>>>(
                d_D, n, kBlockBase, l
            );
            checkKernelErrors();

            int numRowColTiles = numTiles - 1;
            if (numRowColTiles > 0) {
                fwMultiLayerLeadRowKernel<TILE><<<numRowColTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();

                fwMultiLayerLeadColumnKernel<TILE><<<numRowColTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();
            }
        }

        for (int l = blocksInStage - 2; l >= 0; --l) {
            int currentK = kBlockBase + l;

            int numRowTiles = (currentK + 1) + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numRowTiles > 0) {
                fwMultiLayerLeadRowReverseKernel<TILE><<<numRowTiles, blockDim, 2 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }

            int numColTiles = currentK + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numColTiles > 0) {
                fwMultiLayerLeadColumnReverseKernel<TILE><<<numColTiles, blockDim, 2 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }
        }

        if (blocksInStage < numTiles) {
            int numActiveTiles = numTiles - blocksInStage;
            dim3 gridRest(numActiveTiles, numActiveTiles);

            fwMultiLayerRestBlocksKernel<TILE><<<gridRest, blockDim, 2 * tileBytes>>>(
                d_D, n, numTiles, kBlockBase, blocksInStage
            );
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}