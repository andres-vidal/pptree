/**
 * @file BenchmarkParams.hpp
 * @brief Benchmark-specific CLI parameters.
 */
#pragma once

#include <string>

namespace pptree::cli {
  /** @brief Benchmark-specific options. */
  struct BenchmarkParams {
    std::string scenarios_path;
    std::string baseline_path;
    std::string output;
    std::string csv;
    std::string format;
    int iterations    = -1;
    float train_ratio = -1;
  };
}
