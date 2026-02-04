#pragma once

#include "constants.h"

// Optimized multi-layer (generalized) tiled GPU implementation of Floyd-Warshall.
//
// Parameters:
// - d_D: device pointer to distance matrix (n x n, row-major)
// - n: number of vertices
// - tileSize: size of each tile (default 32, max 32 due to thread block limits)
// - kappa: number of tile-layers to group together (default 4)
void fwMultiLayerTilingOptimizedGPU(WeightType* d_D, int n, int tileSize, int kappa);
