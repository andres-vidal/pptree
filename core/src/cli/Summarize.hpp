/**
 * @file Summarize.hpp
 * @brief Summarize subcommand handler: display saved model summary.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace CLI { class App; }

namespace ppforest2::cli {
  /** @brief Register summarize subcommand options on @p app. */
  CLI::App * setup_summarize(CLI::App& app, CLIOptions& params);

  /**
   * @brief Run the summarize subcommand.
   * @return Exit code (0 on success).
   */
  int run_summarize(CLIOptions& params);
}
