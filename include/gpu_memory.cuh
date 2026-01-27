#pragma once

#include "constants.h"

namespace GPUMemory {

// Allocates memory for NxN distance matrix on GPU
// Returns device pointer
WeightType* allocate(int n);

// Copies distance matrix from host to device
void copyToDevice(WeightType* d_graph, const WeightType* h_graph, int n);

// Copies distance matrix from device to host
void copyToHost(WeightType* h_graph, const WeightType* d_graph, int n);

// Frees GPU memory
void free(WeightType* d_graph);

} // namespace GPUMemory
