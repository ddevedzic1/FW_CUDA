#pragma once

#include <string>

namespace TestsRunner {

// Executes tests for a given algorithm (CPU or GPU based on algorithm name)
int executeTests(const std::string& algorithmName, int tileSize);

} // namespace TestsRunner
