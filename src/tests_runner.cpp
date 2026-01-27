#include "tests.h"
#include "utils.h"
#include "fw_baseline_cpu.h"
#include <iostream>
#include <string>
#include <stdexcept>

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <algorithm_name> [tile_size]\n";
    std::cerr << "\n";
    std::cerr << "Arguments:\n";
    std::cerr << "  algorithm_name  Name of the algorithm to test\n";
    std::cerr << "  tile_size       Tile size (optional, default 0)\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  " << programName << " baseline_cpu\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string algorithmName = argv[1];
    int tileSize = 0;

    if (argc == 3) {
        try {
            tileSize = std::stoi(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "ERROR: Invalid TILE_SIZE provided (" << argv[2] << "). Must be an integer.\n";
            return 1;
        }
    }

    AlgorithmFuncCPU algorithm = nullptr;
    try {
        algorithm = getCPUAlgorithmFunc(algorithmName);
    } catch (const std::out_of_range& e) {
        std::cerr << "ERROR: Invalid algorithm name.\n";
        std::cerr << e.what() << "\n";
        return 1;
    }

    AlgorithmFuncCPU reference = fwBaselineCPU;

    std::cout << "Running tests for algorithm: " << algorithmName << "\n";
    runTests(algorithm, reference, algorithmName, tileSize);

    return 0;
}
