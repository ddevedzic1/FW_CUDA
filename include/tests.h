#pragma once

#include "algo_types.h"
#include <string>

// Runs all tests for a given CPU algorithm against a reference implementation
void runTestsCPU(AlgorithmFuncCPU algorithm, AlgorithmFuncCPU reference,
    const std::string& name, int tileSize);

// Runs all tests for a given GPU algorithm against a CPU reference implementation
void runTestsGPU(AlgorithmFuncGPU algorithm, AlgorithmFuncCPU reference,
    const std::string& name, int tileSize, int kappa);
