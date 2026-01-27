#pragma once

#include "constants.h"
#include "algo_types.h"
#include <string>

namespace AlgorithmRunner {

// Executes benchmark for a given algorithm (CPU or GPU based on algorithm name)
int executeBenchmark(
    const std::string& algorithmName,
    int n,
    int tileSize,
    double density,
    unsigned int seed
);

} // namespace AlgorithmRunner
