/**
 * @file BenchmarkParams.hpp
 * @brief Benchmark-specific CLI parameters.
 */
#pragma once

#include <string>
#include <vector>

namespace ppforest2::cli {
  /** @brief Benchmark-specific options. */
  struct BenchmarkParams {
    std::string scenarios_path;
    std::string baseline_path;
    std::vector<std::string> outputs;
    std::string format = "table";
  };
}
