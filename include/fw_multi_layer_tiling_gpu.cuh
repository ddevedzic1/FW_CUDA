#pragma once

#include "constants.h"

// Multi-layer (generalized) tiled GPU implementation of Floyd-Warshall algorithm
// Based on the paper: "Generalized blocked Floyd-Warshall algorithm" by Likhoded & Sipeyko (2019)
//
// Key idea: Group kappa tile-layers together, reducing global memory accesses.
// Instead of writing to global memory every r iterations, we write every (l+1)*r iterations
// for lead blocks, and every kappa*r iterations for rest blocks.
//
// Parameters:
// - d_D: device pointer to distance matrix (n x n, row-major)
// - n: number of vertices
// - tileSize: size of each tile (default 32, max 32 due to thread block limits)
// - kappa: number of tile-layers to group together (default 4)
void fwMultiLayerTilingGPU(WeightType* d_D, int n, int tileSize, int kappa);
