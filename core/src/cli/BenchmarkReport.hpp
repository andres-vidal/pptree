/**
 * @file BenchmarkReport.hpp
 * @brief Benchmark result formatting, comparison, and export.
 */
#pragma once

#include "cli/Benchmark.hpp"
#include <string>
#include <optional>

namespace pptree::bench {
  /**
   * @brief Print a formatted results table to stdout.
   *
   * Without a baseline, prints absolute numbers. With a baseline,
   * adds delta columns (green = faster/less memory, red = slower/more memory).
   *
   * @param current   Results from the current run.
   * @param baseline  Optional baseline results for comparison.
   */
  void print_benchmark_table(
    const SuiteResult&                current,
    const std::optional<SuiteResult>& baseline = std::nullopt);

  /**
   * @brief Write suite results to a JSON file.
   */
  void write_results_json(const SuiteResult& result, const std::string& path);

  /**
   * @brief Write suite results to a CSV file.
   */
  void write_results_csv(const SuiteResult& result, const std::string& path);

  /**
   * @brief Format results as a GitHub-flavored markdown table.
   *
   * Returns a string suitable for posting as a PR comment.
   * With a baseline, includes delta columns with emoji indicators.
   */
  std::string format_benchmark_markdown(
    const SuiteResult&                current,
    const std::optional<SuiteResult>& baseline = std::nullopt);
}
