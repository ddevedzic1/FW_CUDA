#include "tests_runner.h"
#include <iostream>
#include <string>
#include <stdexcept>

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <algorithm_name> [tile_size] [kappa]\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  algorithm_name  Name of the algorithm to test\n";
    std::cerr << "  tile_size       Tile size (optional, default 0)\n";
    std::cerr << "  kappa           Number of tile-layers per stage for multi-layer tiling (optional, default 0)\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << programName << " baseline_cpu\n";
    std::cerr << "  " << programName << " baseline_gpu\n";
    std::cerr << "  " << programName << " multi_layer_tiling_gpu 32 4\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string algorithmName = argv[1];
    int tileSize = 0;

    if (argc >= 3) {
        try {
            tileSize = std::stoi(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid TILE_SIZE provided (" << argv[2] << "). Must be an integer.\n";
            return 1;
        }
    }

    int kappa = 0;

    if (argc == 4) {
        try {
            kappa = std::stoi(argv[3]);
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid KAPPA provided (" << argv[3] << "). Must be an integer.\n";
            return 1;
        }
    }

    return TestsRunner::executeTests(algorithmName, tileSize, kappa);
}
