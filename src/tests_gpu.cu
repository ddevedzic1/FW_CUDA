#include "tests.h"
#include "constants.h"
#include "utils.h"
#include "graph_generator.h"
#include "gpu_memory.cuh"
#include "cuda_utils.cuh"
#include "fw_baseline_cpu.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

static int testsPassed = 0;
static int testsTotal = 0;

static void assertEqual(const std::vector<WeightType>& actual, const std::vector<WeightType>& expected, const std::string& testName) {
    testsTotal++;

    if (actual.size() != expected.size()) {
        std::cerr << "[ FAILED ] " << testName << ": Matrix size mismatch (Expected: " << expected.size()
            << ", Actual: " << actual.size() << ")\n";
        return;
    }

    int n = static_cast<int>(std::round(std::sqrt(actual.size())));

    for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[ FAILED ] " << testName << ": Mismatch at position [" << i / n << "]["
                << i % n << "]. Expected: " << expected[i]
                << ", Actual: " << actual[i] << "\n";
            return;
        }
    }

    std::cout << "[ PASSED ] " << testName << "\n";
    testsPassed++;
}

// Runs GPU algorithm on input and returns result
static std::vector<WeightType> runGPUAlgorithm(AlgorithmFuncGPU algorithm,
    const std::vector<WeightType>& input, int n, int tileSize, int kappa) {

    WeightType* d_graph = GPUMemory::allocate(n);
    GPUMemory::copyToDevice(d_graph, input.data(), n);

    algorithm(d_graph, n, tileSize, kappa);

    std::vector<WeightType> result(input.size());
    GPUMemory::copyToHost(result.data(), d_graph, n);
    GPUMemory::free(d_graph);

    return result;
}

static void runTestGPU(AlgorithmFuncGPU algorithm, const std::vector<WeightType>& input,
    const std::vector<WeightType>& expected, int n, int tileSize, int kappa,
    const std::string& testName) {
    std::vector<WeightType> result = runGPUAlgorithm(algorithm, input, n, tileSize, kappa);
    assertEqual(result, expected, testName);
}

static void runTestAgainstReference(AlgorithmFuncGPU algorithm, AlgorithmFuncCPU reference,
    int n, int tileSize, int kappa, double density, unsigned int seed,
    const std::string& testName) {
    std::vector<WeightType> input = generateGraph1D(n, density, seed);

    std::vector<WeightType> expected = input;
    reference(expected.data(), n, 0);

    std::vector<WeightType> result = runGPUAlgorithm(algorithm, input, n, tileSize, kappa);

    assertEqual(result, expected, testName);
}

static void testSingleNode(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    std::vector<WeightType> input = { 0 };
    std::vector<WeightType> expected = { 0 };
    runTestGPU(algorithm, input, expected, 1, tileSize, kappa, "Single node (1x1)");
}

static void testNoEdges(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    std::vector<WeightType> input = {
        0, INF,
        INF, 0
    };
    runTestGPU(algorithm, input, input, 2, tileSize, kappa, "No edges (2x2)");
}

static void testSimplePath(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    std::vector<WeightType> input = {
        0, 1, INF, 100,
        INF, 0, 1, INF,
        INF, INF, 0, 1,
        INF, INF, INF, 0
    };
    std::vector<WeightType> expected = {
        0, 1, 2, 3,
        INF, 0, 1, 2,
        INF, INF, 0, 1,
        INF, INF, INF, 0
    };
    runTestGPU(algorithm, input, expected, 4, tileSize, kappa, "Simple path optimization (4x4)");
}

static void testDisconnectedComponents(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    std::vector<WeightType> input = {
        0, 1, INF, INF,
        1, 0, INF, INF,
        INF, INF, 0, 2,
        INF, INF, 2, 0
    };
    runTestGPU(algorithm, input, input, 4, tileSize, kappa, "Disconnected components (4x4)");
}

static void testNegativeWeights(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    std::vector<WeightType> input = {
        0, 5, 10,
        0, 0, -1,
        INF, INF, 0
    };
    std::vector<WeightType> expected = {
        0, 5, 4,
        0, 0, -1,
        INF, INF, 0
    };
    runTestGPU(algorithm, input, expected, 3, tileSize, kappa, "Negative weights (3x3)");
}

static void testNegativeCycle(AlgorithmFuncGPU algorithm, int tileSize, int kappa) {
    testsTotal++;

    std::vector<WeightType> input = {
        0, 2, INF,
        INF, 0, 1,
        INF, -3, 0
    };

    std::vector<WeightType> result = runGPUAlgorithm(algorithm, input, 3, tileSize, kappa);

    if (hasNegativeCycle(result.data(), 3)) {
        std::cout << "[ PASSED ] Negative cycle detection (3x3)\n";
        testsPassed++;
    }
    else {
        std::cerr << "[ FAILED ] Negative cycle detection (3x3)\n";
    }
}

static void testLargeGraph(AlgorithmFuncGPU algorithm, AlgorithmFuncCPU reference,
    int n, int tileSize, int kappa, double density, unsigned int seed) {
    std::string testName = "Large graph (N=" + std::to_string(n) + ", density=" + std::to_string(density) + ", seed=" + std::to_string(seed) + ")";
    runTestAgainstReference(algorithm, reference, n, tileSize, kappa, density, seed, testName);
}

void runTestsGPU(AlgorithmFuncGPU algorithm, AlgorithmFuncCPU reference,
    const std::string& name, int tileSize, int kappa) {
    testsPassed = 0;
    testsTotal = 0;

    std::cout << "\n========================================\n";
    std::cout << "TESTING (GPU): " << name << " (tileSize=" << tileSize << ", kappa=" << kappa << ")\n";
    std::cout << "========================================\n";

    testSingleNode(algorithm, tileSize, kappa);
    testNoEdges(algorithm, tileSize, kappa);
    testSimplePath(algorithm, tileSize, kappa);
    testDisconnectedComponents(algorithm, tileSize, kappa);
    testNegativeWeights(algorithm, tileSize, kappa);
    testNegativeCycle(algorithm, tileSize, kappa);

    testLargeGraph(algorithm, reference, 128, tileSize, kappa, 0.9, 42);
    testLargeGraph(algorithm, reference, 256, tileSize, kappa, 0.9, 123);
    testLargeGraph(algorithm, reference, 512, tileSize, kappa, 0.9, 456);
    testLargeGraph(algorithm, reference, 1024, tileSize, kappa, 0.9, 789);

    testLargeGraph(algorithm, reference, 128, tileSize, kappa, 0.1, 101);
    testLargeGraph(algorithm, reference, 256, tileSize, kappa, 0.1, 202);
    testLargeGraph(algorithm, reference, 512, tileSize, kappa, 0.1, 303);
    testLargeGraph(algorithm, reference, 1024, tileSize, kappa, 0.1, 404);

    std::cout << "========================================\n";
    std::cout << "RESULTS: " << testsPassed << "/" << testsTotal << " passed\n";
    std::cout << "========================================\n\n";
}
