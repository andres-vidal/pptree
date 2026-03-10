/**
 * @file CLIOptions.hpp
 * @brief CLI argument parsing, validation, and configuration for pptree.
 *
 * Defines the CLIOptions struct and declares functions to parse,
 * validate, and initialize runtime parameters.
 */
#pragma once

#include <string>

namespace pptree::cli {
  /** @brief Available CLI subcommands. */
  enum class Subcommand { none, train, predict, evaluate, benchmark };

  /**
   * @brief All CLI options and runtime parameters.
   *
   * Fields with -1 or empty defaults are sentinel values meaning
   * "not set by the user" and will be resolved by init_params().
   */
  struct CLIOptions {
    int trees    = 100;
    float lambda = 0.5;
    int threads  = -1;
    int seed     = -1;
    float p_vars = -1;
    int n_vars   = -1;
    std::string vars_input;
    float train_ratio = 0.7;
    int iterations    = 1;
    std::string data_path;
    std::string simulate;
    int rows                  = 1000;
    int cols                  = 10;
    int classes               = 2;
    float sim_mean            = 100.0f;
    float sim_mean_separation = 50.0f;
    float sim_sd              = 10.0f;

    // Convergence and warmup (evaluate + benchmark).
    // See ConvergenceCriteria in Benchmark.hpp for algorithm details.
    bool converge      = true;    ///< Adaptive stopping (default on; -i disables).
    int warmup         = 0;       ///< Warmup iterations discarded before measuring.
    float cv_threshold = 0.05f;   ///< CV target (e.g. 0.05 = stop when std < 5% of mean).
    int min_iterations = 10;      ///< Minimum iterations before checking convergence.
    int stable_window  = 3;       ///< Consecutive checks below threshold before stopping.
    int max_iterations = 200;     ///< Hard upper bound on iterations.

    // Benchmark options
    std::string scenarios_path;
    std::string baseline_path;
    std::string bench_output;
    std::string bench_csv;
    std::string bench_format;
    int bench_iterations = -1;

    Subcommand subcommand = Subcommand::none;
    std::string save_path = "model.json";
    std::string model_path;
    std::string output_path;
    std::string export_path;
    bool quiet      = false;
    bool no_save    = false;
    bool no_metrics = false;
    bool no_color   = false;

    bool used_default_seed    = false;
    bool used_default_threads = false;
    bool used_default_vars    = false;
  };

  /**
   * @brief Warn the user about parameters that are ignored for single-tree training.
   */
  void warn_unused_params(const CLIOptions& params);

  /**
   * @brief Resolve sentinel values in CLIOptions to concrete defaults.
   *
   * @param params     The CLI options to initialize (modified in place).
   * @param total_vars Total number of feature columns (0 to skip vars resolution).
   */
  void init_params(CLIOptions& params, int total_vars = 0);

  /**
   * @brief Parse command-line arguments into a CLIOptions struct.
   *
   * @param argc Argument count from main().
   * @param argv Argument vector from main().
   * @return A populated CLIOptions struct.
   */
  CLIOptions parse_args(int argc, char *argv[]);
}
