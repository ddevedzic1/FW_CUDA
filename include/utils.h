#pragma once

#include "constants.h"
#include "algo_types.h"
#include <string>
#include <map>

// Checks if the distance matrix contains a negative cycle
bool hasNegativeCycle(const WeightType* distMatrix, int n);

// CPU algorithm registry
extern const std::map<std::string, AlgorithmFuncCPU> CPU_ALGORITHMS;

// Returns CPU algorithm function by name, throws std::out_of_range if not found
AlgorithmFuncCPU getCPUAlgorithmFunc(const std::string& algorithmName);
