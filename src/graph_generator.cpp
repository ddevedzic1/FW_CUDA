#include "graph_generator.h"
#include "constants.h"
#include <iostream>
#include <random>

// NOTE: Caller must use _aligned_free() to deallocate returned pointer!
WeightType* generateAlignedGraph1D(int n, double density, unsigned int seed)
{
    if (n <= 0 || n > MAX_N) {
        std::cerr << "ERROR: Graph size (n) must be between 1 and " << MAX_N << ". Graph generation failed.\n";
        return nullptr;
    }

    size_t size_n = static_cast<size_t>(n);
    size_t total_elements = size_n * size_n;
    size_t total_bytes = total_elements * sizeof(WeightType);

    WeightType* D = static_cast<WeightType*>(ALIGNED_MALLOC(total_bytes, 32));

    if (D == nullptr) {
        std::cerr << "ERROR: Aligned memory allocation failed.\n";
        return nullptr;
    }

    std::mt19937 generator(seed != 0 ? seed : std::random_device{}());
    std::uniform_int_distribution<WeightType> weight_dist(0, MAX_EDGE_WEIGHT);
    std::bernoulli_distribution density_dist(density);

    for (size_t i = 0; i < size_n; ++i) {
        for (size_t j = 0; j < size_n; ++j) {
            size_t index = i * size_n + j;

            if (i == j) {
                D[index] = 0;
            }
            else if (density_dist(generator)) {
                D[index] = weight_dist(generator);
            }
            else {
                D[index] = INF;
            }
        }
    }

    return D;
}

std::vector<WeightType> generateGraph1D(int n, double density, unsigned int seed) {
    if (n <= 0 || n > MAX_N) {
        std::cerr << "ERROR: Graph size (n) must be between 1 and " << MAX_N << ". Graph generation failed.\n";
        return {};
    }

    size_t size_n = static_cast<size_t>(n);
    size_t total_elements = size_n * size_n;

    std::vector<WeightType> D(total_elements);

    std::mt19937 generator(seed != 0 ? seed : std::random_device{}());
    std::uniform_int_distribution<WeightType> weight_dist(0, MAX_EDGE_WEIGHT);
    std::bernoulli_distribution density_dist(density);

    for (size_t i = 0; i < size_n; ++i) {
        for (size_t j = 0; j < size_n; ++j) {
            size_t index = i * size_n + j;

            if (i == j) {
                D[index] = 0;
            }
            else if (density_dist(generator)) {
                D[index] = weight_dist(generator);
            }
            else {
                D[index] = INF;
            }
        }
    }

    return D;
}