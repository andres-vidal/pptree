/**
 * @file Evaluate.hpp
 * @brief Multi-iteration model evaluation with convergence and the
 *        evaluate subcommand handler.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register evaluate subcommand options on @p app. */
  void setup_evaluate(CLI::App& app, Params& params);

  /** @brief Add evaluate/convergence options shared by evaluate and benchmark. */
  void add_evaluate_options(CLI::App* sub, EvaluateParams& evaluate);

  /**
   * @brief Run the evaluate subcommand.
   * @return Exit code (0 on success).
   */
  int run_evaluate(Params& params);
}
