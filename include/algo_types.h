#pragma once

#include "constants.h"

// CPU algorithm function pointer type
using AlgorithmFuncCPU = void(*)(WeightType* D, int n, int tileSize);

// GPU algorithm function pointer type
using AlgorithmFuncGPU = void(*)(WeightType* d_D, int n, int tileSize);