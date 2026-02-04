#include "algorithm_runner.h"
#include "constants.h"
#include <iostream>
#include <string>
#include <stdexcept>

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <algorithm_name> <n_size> <density> [seed] [tile_size] [kappa]\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  algorithm_name  Name of the algorithm to run\n";
    std::cerr << "  n_size          Graph size (number of vertices, 1 to " << MAX_N << ")\n";
    std::cerr << "  density         Edge density (0.0 to 1.0)\n";
    std::cerr << "  seed            Random seed (optional, 0 for random)\n";
    std::cerr << "  tile_size       Tile size for tiling/CUDA (optional)\n";
    std::cerr << "  kappa           Number of tile-layers to group (optional, used by multi_layer_tiling_gpu)\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << programName << " baseline_cpu 1024 0.8\n";
    std::cerr << "  " << programName << " baseline_gpu 2048 0.9 42 16\n";
    std::cerr << "  " << programName << " multi_layer_tiling_gpu 4096 0.9 42 32 4\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 7) {
        printUsage(argv[0]);
        return 1;
    }

    std::string algorithmName = argv[1];
    int n = 0;
    double density = 0.0;
    unsigned int seed = 0;
    int tileSize = 0;
    int kappa = 0;

    try {
        n = std::stoi(argv[2]);
        if (n <= 0 || n > MAX_N) {
            std::cerr << "ERROR: Invalid graph size (" << n << "). Must be between 1 and " << MAX_N << ".\n";
            return 1;
        }
    } catch (const std::exception&) {
        std::cerr << "ERROR: Invalid N_SIZE provided (" << argv[2] << "). Must be an integer.\n";
        return 1;
    }

    try {
        density = std::stod(argv[3]);
        if (density < 0.0 || density > 1.0) {
            std::cerr << "ERROR: Invalid density (" << density << "). Must be between 0.0 and 1.0.\n";
            return 1;
        }
    } catch (const std::exception&) {
        std::cerr << "ERROR: Invalid DENSITY provided (" << argv[3] << "). Must be a floating point number.\n";
        return 1;
    }

    if (argc >= 5) {
        try {
            seed = std::stoul(argv[4]);
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid SEED provided (" << argv[4] << "). Must be an unsigned integer.\n";
            return 1;
        }
    }

    if (argc >= 6) {
        try {
            tileSize = std::stoi(argv[5]);
            if (tileSize < 0 || tileSize > n) {
                std::cerr << "ERROR: Invalid TILE_SIZE (" << tileSize << "). Must be between 0 and " << n << ".\n";
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid TILE_SIZE provided (" << argv[5] << "). Must be an integer.\n";
            return 1;
        }
    }

    if (argc == 7) {
        try {
            kappa = std::stoi(argv[6]);
            if (kappa < 0) {
                std::cerr << "ERROR: Invalid KAPPA (" << kappa << "). Must be non-negative.\n";
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid KAPPA provided (" << argv[6] << "). Must be an integer.\n";
            return 1;
        }
    }

    try {
        return AlgorithmRunner::executeBenchmark(algorithmName, n, tileSize, kappa, density, seed);
    } catch (const std::exception& e) {
        std::cerr << "RUNTIME ERROR: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "RUNTIME ERROR: An unknown error occurred.\n";
        return 1;
    }
}
