#pragma once

#include "constants.h"

// Tiled GPU implementation of Floyd-Warshall algorithm using shared memory
void fwTilingGPU(WeightType* d_D, int n, int tileSize, int kappa);
