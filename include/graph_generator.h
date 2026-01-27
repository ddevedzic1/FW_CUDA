#pragma once

#include <vector>
#include <cstdlib>
#include "constants.h"

// Platform-independent aligned memory allocation
#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_MALLOC(size, align) _aligned_malloc(size, align)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    inline void* aligned_malloc_portable(size_t size, size_t align) {
        void* ptr = nullptr;
        posix_memalign(&ptr, align, size);
        return ptr;
    }
    #define ALIGNED_MALLOC(size, align) aligned_malloc_portable(size, align)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// Generates a random graph as a 1D array with aligned memory allocation
// NOTE: Caller must use ALIGNED_FREE() to deallocate returned pointer!
WeightType* generateAlignedGraph1D(int n, double density, unsigned int seed = 0);

// Generates a random graph as a std::vector (for testing convenience)
std::vector<WeightType> generateGraph1D(int n, double density, unsigned int seed = 0);