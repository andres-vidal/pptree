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
  CLI::App* setup_evaluate(CLI::App& app, CLIOptions& params);

  /**
   * @brief Run the evaluate subcommand.
   * @return Exit code (0 on success).
   */
  int run_evaluate(CLIOptions& params);
}
