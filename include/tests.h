#pragma once

#include "algo_types.h"
#include <string>

// Runs all tests for a given algorithm against a reference implementation
void runTests(AlgorithmFuncCPU algorithm, AlgorithmFuncCPU reference,
    const std::string& name, int tileSize);
