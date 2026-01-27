#include "tests.h"
#include "constants.h"
#include "utils.h"
#include "graph_generator.h"
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

static void runTest(AlgorithmFuncCPU algorithm, const std::vector<WeightType>& input,
    const std::vector<WeightType>& expected, int n, int tileSize,
    const std::string& testName) {
    std::vector<WeightType> result = input;
    algorithm(result.data(), n, tileSize);
    assertEqual(result, expected, testName);
}

static void runTestAgainstReference(AlgorithmFuncCPU algorithm, AlgorithmFuncCPU reference,
    int n, int tileSize, double density, unsigned int seed,
    const std::string& testName) {
    std::vector<WeightType> input = generateGraph1D(n, density, seed);

    std::vector<WeightType> expected = input;
    reference(expected.data(), n, 0);

    std::vector<WeightType> result = input;
    algorithm(result.data(), n, tileSize);

    assertEqual(result, expected, testName);
}

static void testSingleNode(AlgorithmFuncCPU algorithm, int tileSize) {
    std::vector<WeightType> input = { 0 };
    std::vector<WeightType> expected = { 0 };
    runTest(algorithm, input, expected, 1, tileSize, "Single node (1x1)");
}

static void testNoEdges(AlgorithmFuncCPU algorithm, int tileSize) {
    std::vector<WeightType> input = {
        0, INF,
        INF, 0
    };
    runTest(algorithm, input, input, 2, tileSize, "No edges (2x2)");
}

static void testSimplePath(AlgorithmFuncCPU algorithm, int tileSize) {
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
    runTest(algorithm, input, expected, 4, tileSize, "Simple path optimization (4x4)");
}

static void testDisconnectedComponents(AlgorithmFuncCPU algorithm, int tileSize) {
    std::vector<WeightType> input = {
        0, 1, INF, INF,
        1, 0, INF, INF,
        INF, INF, 0, 2,
        INF, INF, 2, 0
    };
    runTest(algorithm, input, input, 4, tileSize, "Disconnected components (4x4)");
}

static void testNegativeWeights(AlgorithmFuncCPU algorithm, int tileSize) {
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
    runTest(algorithm, input, expected, 3, tileSize, "Negative weights (3x3)");
}

static void testNegativeCycle(AlgorithmFuncCPU algorithm, int tileSize) {
    testsTotal++;

    std::vector<WeightType> input = {
        0, 2, INF,
        INF, 0, 1,
        INF, -3, 0
    };

    std::vector<WeightType> result = input;
    algorithm(result.data(), 3, tileSize);

    if (hasNegativeCycle(result.data(), 3)) {
        std::cout << "[ PASSED ] Negative cycle detection (3x3)\n";
        testsPassed++;
    }
    else {
        std::cerr << "[ FAILED ] Negative cycle detection (3x3)\n";
    }
}

static void testLargeGraph(AlgorithmFuncCPU algorithm, AlgorithmFuncCPU reference,
    int n, int tileSize, unsigned int seed) {
    std::string testName = "Large graph (N=" + std::to_string(n) + ", seed=" + std::to_string(seed) + ")";
    runTestAgainstReference(algorithm, reference, n, tileSize, 0.9, seed, testName);
}

void runTests(AlgorithmFuncCPU algorithm, AlgorithmFuncCPU reference,
    const std::string& name, int tileSize) {
    testsPassed = 0;
    testsTotal = 0;

    std::cout << "\n========================================\n";
    std::cout << "TESTING: " << name << " (tileSize=" << tileSize << ")\n";
    std::cout << "========================================\n";

    testSingleNode(algorithm, tileSize);
    testNoEdges(algorithm, tileSize);
    testSimplePath(algorithm, tileSize);
    testDisconnectedComponents(algorithm, tileSize);
    testNegativeWeights(algorithm, tileSize);
    testNegativeCycle(algorithm, tileSize);

    testLargeGraph(algorithm, reference, 128, tileSize, 42);
    testLargeGraph(algorithm, reference, 256, tileSize, 123);
    testLargeGraph(algorithm, reference, 512, tileSize, 456);
    testLargeGraph(algorithm, reference, 1024, tileSize, 789);

    std::cout << "========================================\n";
    std::cout << "RESULTS: " << testsPassed << "/" << testsTotal << " passed\n";
    std::cout << "========================================\n\n";
}
