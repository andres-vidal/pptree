/**
 * @file EvaluateParams.hpp
 * @brief Evaluate-specific CLI parameters: simulation, convergence, and evaluation options.
 */
#pragma once

#include <string>

namespace ppforest2::cli {
  /** @brief Simulation data source options. */
  struct SimulateParams {
    std::string format;
    int rows              = 1000;
    int cols              = 10;
    int n_groups          = 2;
    float mean            = 100.0f;
    float mean_separation = 50.0f;
    float sd              = 10.0f;
  };

  /**
   * @brief Convergence and iteration control (evaluate + benchmark).
   *
   * See ConvergenceCriteria in Benchmark.hpp for the per-scenario
   * equivalent used when parsing benchmark JSON files.
   */
  struct ConvergenceParams {
    bool enabled = true;    ///< Adaptive stopping (default on; -i disables).
    int warmup   = 0;       ///< Warmup iterations discarded before measuring.
    float cv     = 0.05f;   ///< CV target (e.g. 0.05 = stop when std < 5% of mean).
    int min      = 10;      ///< Minimum iterations before checking convergence.
    int window   = 3;       ///< Consecutive checks below threshold before stopping.
    int max      = 200;     ///< Hard upper bound on iterations.
  };

  /** @brief Evaluate-specific options. */
  struct EvaluateParams {
    float train_ratio = 0.7;
    int iterations    = 1;
    std::string export_path;
  };
}
