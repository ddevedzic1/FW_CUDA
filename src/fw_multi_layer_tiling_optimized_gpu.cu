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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDiag  = sharedMem;
    WeightType* sharedRow[2] = { sharedMem +     TILE * TILE, sharedMem + 2 * TILE * TILE };
    WeightType* sharedCol[2] = { sharedMem + 3 * TILE * TILE, sharedMem + 4 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int cur = 0;
    if (l > 0) {
        int mBase = kBlockBase * TILE;
        sharedRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = 0; m < l; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < l) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedRow[cur][r0 * TILE + k];
            WeightType row1 = sharedRow[cur][r1 * TILE + k];
            WeightType col0 = sharedCol[cur][k * TILE + c0];
            WeightType col1 = sharedCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = colBase  + c0;
    int gJ1 = colBase  + c1;

    WeightType* sharedDiag     = sharedMem;
    WeightType* sharedCurr     = sharedMem +     TILE * TILE;
    WeightType* sharedDDRow[2] = { sharedMem + 2 * TILE * TILE, sharedMem + 3 * TILE * TILE };
    WeightType* sharedDDCol[2] = { sharedMem + 4 * TILE * TILE, sharedMem + 5 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int mStart = (colTile >= kBlockBase && colTile < currentK)
        ? (colTile - kBlockBase + 1) : 0;

    int cur = 0;
    if (mStart < l) {
        int mBase = (kBlockBase + mStart) * TILE;
        sharedDDRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedDDCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = mStart; m < l; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < l) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedDDRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedDDRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedDDCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedDDCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedDDCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedDDCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[cur][r0 * TILE + k];
            WeightType row1 = sharedDDRow[cur][r1 * TILE + k];
            WeightType col0 = sharedDDCol[cur][k * TILE + c0];
            WeightType col1 = sharedDDCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowBase  + r0;
    int gI1 = rowBase  + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDiag     = sharedMem;
    WeightType* sharedCurr     = sharedMem +     TILE * TILE;
    WeightType* sharedDDRow[2] = { sharedMem + 2 * TILE * TILE, sharedMem + 3 * TILE * TILE };
    WeightType* sharedDDCol[2] = { sharedMem + 4 * TILE * TILE, sharedMem + 5 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int mStart = (rowTile >= kBlockBase && rowTile < currentK)
        ? (rowTile - kBlockBase + 1) : 0;

    int cur = 0;
    if (mStart < l) {
        int mBase = (kBlockBase + mStart) * TILE;
        sharedDDRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedDDCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = mStart; m < l; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < l) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedDDRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedDDRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedDDCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedDDCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedDDCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedDDCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[cur][r0 * TILE + k];
            WeightType row1 = sharedDDRow[cur][r1 * TILE + k];
            WeightType col0 = sharedDDCol[cur][k * TILE + c0];
            WeightType col1 = sharedDDCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = diagBase + r0;
    int gI1 = diagBase + r1;
    int gJ0 = colBase  + c0;
    int gJ1 = colBase  + c1;

    WeightType* sharedDDRow[2] = { sharedMem, sharedMem + TILE * TILE };
    WeightType* sharedDDCol[2] = { sharedMem + 2 * TILE * TILE, sharedMem + 3 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int cur = 0;
    if (l + 1 < blocksInStage) {
        int mBase = (kBlockBase + l + 1) * TILE;
        sharedDDRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedDDCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = l + 1; m < blocksInStage; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < blocksInStage) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedDDRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedDDRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedDDCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedDDCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedDDCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedDDCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[cur][r0 * TILE + k];
            WeightType row1 = sharedDDRow[cur][r1 * TILE + k];
            WeightType col0 = sharedDDCol[cur][k * TILE + c0];
            WeightType col1 = sharedDDCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
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
    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowBase  + r0;
    int gI1 = rowBase  + r1;
    int gJ0 = diagBase + c0;
    int gJ1 = diagBase + c1;

    WeightType* sharedDDRow[2] = { sharedMem, sharedMem + TILE * TILE };
    WeightType* sharedDDCol[2] = { sharedMem + 2 * TILE * TILE, sharedMem + 3 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int cur = 0;
    if (l + 1 < blocksInStage) {
        int mBase = (kBlockBase + l + 1) * TILE;
        sharedDDRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedDDRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedDDRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedDDRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedDDCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedDDCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedDDCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedDDCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = l + 1; m < blocksInStage; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < blocksInStage) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedDDRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedDDRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedDDRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedDDCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedDDCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedDDCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedDDCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedDDRow[cur][r0 * TILE + k];
            WeightType row1 = sharedDDRow[cur][r1 * TILE + k];
            WeightType col0 = sharedDDCol[cur][k * TILE + c0];
            WeightType col1 = sharedDDCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
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

    int r0 = ty * 2;
    int r1 = r0 + 1;
    int c0 = tx * 2;
    int c1 = c0 + 1;
    int gI0 = rowTile * TILE + r0;
    int gI1 = rowTile * TILE + r1;
    int gJ0 = colTile * TILE + c0;
    int gJ1 = colTile * TILE + c1;

    WeightType* sharedRow[2] = { sharedMem, sharedMem + TILE * TILE };
    WeightType* sharedCol[2] = { sharedMem + 2 * TILE * TILE, sharedMem + 3 * TILE * TILE };

    WeightType c00 = (gI0 < n && gJ0 < n) ? __ldg(&D[gI0 * n + gJ0]) : INF;
    WeightType c01 = (gI0 < n && gJ1 < n) ? __ldg(&D[gI0 * n + gJ1]) : INF;
    WeightType c10 = (gI1 < n && gJ0 < n) ? __ldg(&D[gI1 * n + gJ0]) : INF;
    WeightType c11 = (gI1 < n && gJ1 < n) ? __ldg(&D[gI1 * n + gJ1]) : INF;

    int cur = 0;
    if (blocksInStage > 0) {
        int mBase = kBlockBase * TILE;
        sharedRow[0][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
        sharedRow[0][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
        sharedRow[0][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
        sharedRow[0][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
        sharedCol[0][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
        sharedCol[0][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
        sharedCol[0][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
        sharedCol[0][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        __syncthreads();
    }

    for (int m = 0; m < blocksInStage; ++m) {
        int nxt = 1 - cur;
        if (m + 1 < blocksInStage) {
            int mBase = (kBlockBase + m + 1) * TILE;
            sharedRow[nxt][r0 * TILE + c0] = (gI0 < n && mBase + c0 < n) ? __ldg(&D[gI0 * n + mBase + c0]) : INF;
            sharedRow[nxt][r0 * TILE + c1] = (gI0 < n && mBase + c1 < n) ? __ldg(&D[gI0 * n + mBase + c1]) : INF;
            sharedRow[nxt][r1 * TILE + c0] = (gI1 < n && mBase + c0 < n) ? __ldg(&D[gI1 * n + mBase + c0]) : INF;
            sharedRow[nxt][r1 * TILE + c1] = (gI1 < n && mBase + c1 < n) ? __ldg(&D[gI1 * n + mBase + c1]) : INF;
            sharedCol[nxt][r0 * TILE + c0] = (mBase + r0 < n && gJ0 < n) ? __ldg(&D[(mBase + r0) * n + gJ0]) : INF;
            sharedCol[nxt][r0 * TILE + c1] = (mBase + r0 < n && gJ1 < n) ? __ldg(&D[(mBase + r0) * n + gJ1]) : INF;
            sharedCol[nxt][r1 * TILE + c0] = (mBase + r1 < n && gJ0 < n) ? __ldg(&D[(mBase + r1) * n + gJ0]) : INF;
            sharedCol[nxt][r1 * TILE + c1] = (mBase + r1 < n && gJ1 < n) ? __ldg(&D[(mBase + r1) * n + gJ1]) : INF;
        }

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            WeightType row0 = sharedRow[cur][r0 * TILE + k];
            WeightType row1 = sharedRow[cur][r1 * TILE + k];
            WeightType col0 = sharedCol[cur][k * TILE + c0];
            WeightType col1 = sharedCol[cur][k * TILE + c1];
            c00 = min(c00, row0 + col0);
            c01 = min(c01, row0 + col1);
            c10 = min(c10, row1 + col0);
            c11 = min(c11, row1 + col1);
        }

        __syncthreads();
        cur = nxt;
    }

    if (gI0 < n && gJ0 < n) D[gI0 * n + gJ0] = c00;
    if (gI0 < n && gJ1 < n) D[gI0 * n + gJ1] = c01;
    if (gI1 < n && gJ0 < n) D[gI1 * n + gJ0] = c10;
    if (gI1 < n && gJ1 < n) D[gI1 * n + gJ1] = c11;
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

    dim3 blockDim(TILE / 2, TILE / 2);
    size_t tileBytes = static_cast<size_t>(TILE) * TILE * sizeof(WeightType);

    for (int kBlockBase = 0; kBlockBase < numTiles; kBlockBase += kappa) {
        int blocksInStage = min(kappa, numTiles - kBlockBase);

        for (int l = 0; l < blocksInStage; ++l) {
            fwMLOptLeadBlockKernel<TILE><<<1, blockDim, 5 * tileBytes>>>(
                d_D, n, kBlockBase, l
            );
            checkKernelErrors();

            int numRowColTiles = numTiles - 1;
            if (numRowColTiles > 0) {
                fwMLOptLeadRowKernel<TILE><<<numRowColTiles, blockDim, 6 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();

                fwMLOptLeadColumnKernel<TILE><<<numRowColTiles, blockDim, 6 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, l
                );
                checkKernelErrors();
            }
        }

        for (int l = blocksInStage - 2; l >= 0; --l) {
            int currentK = kBlockBase + l;

            int numRowTiles = (currentK + 1) + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numRowTiles > 0) {
                fwMLOptLeadRowReverseKernel<TILE><<<numRowTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }

            int numColTiles = currentK + max(0, numTiles - (kBlockBase + blocksInStage));
            if (numColTiles > 0) {
                fwMLOptLeadColumnReverseKernel<TILE><<<numColTiles, blockDim, 4 * tileBytes>>>(
                    d_D, n, numTiles, kBlockBase, blocksInStage, l
                );
                checkKernelErrors();
            }
        }

        if (blocksInStage < numTiles) {
            int numActiveTiles = numTiles - blocksInStage;
            dim3 gridRest(numActiveTiles, numActiveTiles);

            fwMLOptRestBlocksKernel<TILE><<<gridRest, blockDim, 4 * tileBytes>>>(
                d_D, n, numTiles, kBlockBase, blocksInStage
            );
            checkKernelErrors();
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}
