/**
 * @file BenchmarkReport.hpp
 * @brief Benchmark result formatting, comparison, and export.
 */
#pragma once

#include "cli/Benchmark.hpp"
#include "io/Output.hpp"
#include <string>
#include <optional>

namespace ppforest2::cli {
  /**
   * @brief Print a formatted results table to stdout.
   *
   * Without a baseline, prints absolute numbers. With a baseline,
   * adds delta columns (green = faster/less memory, red = slower/more memory).
   *
   * @param out       Output context.
   * @param current   Results from the current run.
   * @param baseline  Optional baseline results for comparison.
   */
  void print_benchmark_table(
      io::Output& out, SuiteResult const& current, std::optional<SuiteResult> const& baseline = std::nullopt
  );

  /**
   * @brief Write suite results to a JSON file.
   */
  void write_results_json(SuiteResult const& result, std::string const& path);

  /**
   * @brief Write suite results to a CSV file.
   */
  void write_results_csv(SuiteResult const& result, std::string const& path);

  /**
   * @brief Print results as a GitHub-flavored markdown table.
   *
   * Prints directly via Output (no indentation). Suitable for
   * piping to a file or posting as a PR comment.
   * With a baseline, includes delta columns with emoji indicators.
   *
   * @param out       Output context.
   * @param current   Results from the current run.
   * @param baseline  Optional baseline results for comparison.
   */
  void print_benchmark_markdown(
      io::Output& out, SuiteResult const& current, std::optional<SuiteResult> const& baseline = std::nullopt
  );
}
