#pragma once

#include "constants.h"

// Basic CPU implementation of Floyd-Warshall algorithm (reference implementation)
void fwBaselineCPU(WeightType* D, int n, int tileSize);
