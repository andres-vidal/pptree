/**
 * @file CLIOptions.hpp
 * @brief CLI argument parsing, validation, and configuration for ppforest2.
 *
 * Defines the CLIOptions struct and declares functions to parse,
 * validate, and initialize runtime parameters.
 */
#pragma once

#include "cli/ModelParams.hpp"
#include "cli/EvaluateParams.hpp"
#include "cli/BenchmarkParams.hpp"
#include "io/Output.hpp"

#include <nlohmann/json.hpp>
#include <string>

/**
 * @brief Command-line interface: argument parsing, subcommands, and
 *        benchmark/evaluation orchestration.
 */
namespace ppforest2::cli {
  /** @brief Available CLI subcommands. */
  enum class Subcommand { none, train, predict, evaluate, benchmark, summarize };

  /**
   * @brief All CLI options and runtime parameters.
   *
   * Fields with -1 or empty defaults are sentinel values meaning
   * "not set by the user" and will be resolved by init_params().
   */
  struct CLIOptions {
    Subcommand subcommand = Subcommand::none;

    ModelParams model;
    SimulateParams simulation;
    ConvergenceParams convergence;
    EvaluateParams evaluate;
    BenchmarkParams benchmark;

    std::string data_path;
    std::string save_path = "model.json";
    std::string model_path;
    std::string output_path;

    bool quiet          = false;
    bool no_save        = false;
    bool no_metrics     = false;
    bool no_color       = false;
    bool no_proportions = false;

    /** @brief Path to JSON config file (--config). */
    std::string config_path;
  };

  /**
   * @brief Warn the user about parameters that are ignored for single-tree training.
   */
  void warn_unused_params(io::Output& out, CLIOptions const& params);

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
  CLIOptions parse_args(int argc, char* argv[]);
}
