/**
 * @file Predict.hpp
 * @brief Predict subcommand handler.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace CLI { class App; }

namespace pptree::cli {
  /** @brief Register predict subcommand options on @p app. */
  CLI::App *setup_predict(CLI::App& app, CLIOptions& params);

  /**
   * @brief Run the predict subcommand.
   * @return Exit code (0 on success).
   */
  int run_predict(CLIOptions& params);
}
