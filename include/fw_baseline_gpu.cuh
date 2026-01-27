#pragma once

#include "constants.h"

// Basic GPU implementation of Floyd-Warshall algorithm
void fwBaselineGPU(WeightType* d_D, int n, int tileSize);
