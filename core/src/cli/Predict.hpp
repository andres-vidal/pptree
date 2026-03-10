/**
 * @file Predict.hpp
 * @brief Predict subcommand handler.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace pptree::cli {
  /**
   * @brief Run the predict subcommand.
   * @return Exit code (0 on success).
   */
  int run_predict(CLIOptions& params);
}
