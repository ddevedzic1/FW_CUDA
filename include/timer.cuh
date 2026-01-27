#pragma once

#include "constants.h"
#include "algo_types.h"

namespace Timer {

// Performs GPU warmup to eliminate first-launch overhead from measurements
void warmupGPU();

// Measures CPU algorithm execution time in seconds
double measureCPU(
    AlgorithmFuncCPU func,
    WeightType* graph,
    int n,
    int tileSize,
    int warmup_runs = 0
);

// Measures GPU algorithm execution time using CUDA events, returns time in seconds
double measureGPU(
    AlgorithmFuncGPU func,
    WeightType* d_graph,
    int n,
    int tileSize,
    int warmup_runs = 1
);

} // namespace Timer
