/**
 * @file Benchmark.hpp
 * @brief Benchmark scenario types, JSON parsing, and subprocess-based execution.
 *
 * Each benchmark scenario runs as a separate `ppforest2 evaluate` process,
 * giving accurate per-scenario peak RSS measurements.
 */
#pragma once

#include "cli/CLIOptions.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register benchmark subcommand options on @p app. */
  CLI::App* setup_benchmark(CLI::App& app, CLIOptions& params);

  /**
   * @brief Convergence criteria for adaptive stopping in benchmarks.
   *
   * Instead of running a fixed number of iterations, convergence mode
   * monitors the coefficient of variation (CV = std/mean) of timing
   * measurements and stops once results are statistically stable.
   *
   * The algorithm works as follows:
   * 1. After each iteration, compute CV across all timing samples.
   * 2. If CV < cv_threshold, increment a stability counter.
   *    Otherwise, reset the counter to zero.
   * 3. Stop when the counter reaches stable_window (i.e., CV stayed
   *    below threshold for stable_window consecutive iterations).
   * 4. Never check before min_iterations (need enough samples).
   * 5. Never exceed max_iterations (hard cap to prevent runaway).
   */
  struct ConvergenceCriteria {
    /** Target CV threshold (e.g., 0.05 = stop when std < 5% of mean). */
    float cv = 0.05f;
    /** Number of consecutive iterations that must stay below the CV threshold. */
    int window = 3;
    /** Minimum iterations before convergence checks begin. */
    int min = 10;
    /** Hard upper bound on iterations (stops even if not converged). */
    int max = 200;
  };

  /**
   * @brief A single benchmark scenario: data shape + model config.
   */
  struct Scenario {
    std::string name;

    // Data source: either a CSV path or simulated NxPxG
    std::string data;

    // Data parameters (used for simulation; ignored when data is set)
    int n = 1000;
    int p = 10;
    int g = 3;

    // Model parameters
    int size     = 100;
    float p_vars = 0.5F;
    float lambda = 0.5F;
    int threads  = -1;

    // Strategy configs (optional; passed as --pp, --vars, etc. to evaluate)
    nlohmann::json pp_config;
    nlohmann::json vars_config;
    nlohmann::json cutpoint_config;
    nlohmann::json stop_config;
    nlohmann::json binarize_config;
    nlohmann::json partition_config;

    // Evaluation parameters
    float train_ratio = 0.7F;
    int seed          = 0;
    int warmup        = 0;

    // Iteration mode: if iterations > 0, fixed mode; otherwise convergence
    int iterations = -1;
    ConvergenceCriteria convergence;
  };

  /**
   * @brief Result of running a single benchmark scenario.
   *
   * Parsed from the JSON output of `ppforest2 evaluate`.
   */
  struct ScenarioResult {
    std::string name;
    std::string data;

    // Data shape (copied from scenario for reporting; 0 for real-data scenarios)
    int n = 0, p = 0, g = 0;
    int size          = 0;
    float p_vars      = 0;
    float train_ratio = 0.7F;

    // Aggregated metrics
    int runs             = 0;
    double mean_time_ms  = 0;
    double std_time_ms   = 0;
    double mean_tr_error = 0;
    double mean_te_error = 0;
    long peak_rss_bytes  = -1;
    double peak_rss_mb   = -1;

    // Scenario wall-clock time (including warmup + process overhead)
    double scenario_time_ms = 0;
  };

  /**
   * @brief A suite of scenarios with shared defaults.
   */
  struct BenchmarkSuite {
    std::string name = "ppforest2 benchmark";
    std::vector<Scenario> scenarios;
  };

  /**
   * @brief Aggregated results for an entire suite run.
   */
  struct SuiteResult {
    std::string suite_name;
    std::string timestamp;
    std::vector<ScenarioResult> results;
    double total_time_ms = 0;

    nlohmann::json to_json() const;
  };

  /**
   * @brief Parse a BenchmarkSuite from a JSON file path.
   * @throws std::runtime_error on parse or validation errors.
   */
  BenchmarkSuite parse_suite(std::string const& path);

  /**
   * @brief Parse a BenchmarkSuite from a JSON object.
   * @throws std::runtime_error on validation errors.
   */
  BenchmarkSuite parse_suite(nlohmann::json const& j);

  /**
   * @brief Parse a SuiteResult from a JSON file (for baseline comparison).
   */
  SuiteResult parse_results(std::string const& path);

  /**
   * @brief Callback for progress reporting during benchmark execution.
   * @param scenario_index  Current scenario index (0-based).
   * @param total           Total number of scenarios.
   * @param name            Name of the current scenario.
   */
  using ProgressCallback = std::function<void(int scenario_index, int total, std::string const& name)>;

  /**
   * @brief Run all scenarios in a suite via subprocess invocations.
   *
   * For each scenario, spawns `ppforest2 evaluate` as a child process with
   * the appropriate flags, reads its JSON output, and collects results.
   *
   * @param suite       The benchmark suite to run.
   * @param binary_path Path to the ppforest2 binary (typically argv[0]).
   * @param quiet       Suppress progress output.
   * @param progress    Optional progress callback.
   * @return Aggregated suite results.
   */
  SuiteResult run_suite(
      BenchmarkSuite const& suite,
      std::string const& binary_path,
      bool quiet                = false,
      ProgressCallback progress = nullptr
  );

  /**
   * @brief Run a single scenario as a subprocess.
   */
  ScenarioResult run_scenario(Scenario const& scenario, std::string const& binary_path, bool quiet = false);

  /**
   * @brief Run the benchmark subcommand.
   * @param params      CLI options.
   * @param binary_path Path to the ppforest2 binary (typically argv[0]).
   * @return Exit code.
   */
  int run_benchmark(CLIOptions& params, std::string const& binary_path);
}
