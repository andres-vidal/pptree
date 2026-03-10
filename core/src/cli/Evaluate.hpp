/**
 * @file Evaluate.hpp
 * @brief Multi-iteration model evaluation with convergence and the
 *        evaluate subcommand handler.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace pptree::cli {
  /**
   * @brief Run the evaluate subcommand.
   * @return Exit code (0 on success).
   */
  int run_evaluate(CLIOptions& params);
}
