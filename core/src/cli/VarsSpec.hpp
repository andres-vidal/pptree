/**
 * @file VarsSpec.hpp
 * @brief Shared parsing for the --vars / vars parameter.
 *
 * Supports multiple input formats:
 * - Integer count: 5 (number of features per split)
 * - Decimal proportion: 0.5 (fraction of total features)
 * - String fraction: "1/3" (computed proportion)
 *
 * Used by both CLI argument parsing (CLIOptions.hpp) and
 * benchmark scenario parsing (Benchmark.cpp).
 */
#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::cli {
  /**
   * @brief Result of parsing a vars specification.
   *
   * Distinguishes between a proportion (ready to use) and an integer
   * count (needs conversion using the total number of features).
   */
  struct VarsSpec {
    bool is_proportion;
    float value;
  };

  /**
   * @brief Parse vars from a string (CLI input).
   *
   * Formats:
   * - "1/3"  → fraction, is_proportion = true, value = 0.333...
   * - "0.5"  → decimal, is_proportion = true, value = 0.5
   * - "5"    → count,   is_proportion = false, value = 5
   *
   * @throws std::runtime_error on invalid input.
   */
  VarsSpec parse_vars(const std::string& input);

  /**
   * @brief Parse vars from a JSON value (benchmark scenarios).
   *
   * Formats:
   * - JSON string: delegates to parse_vars(string)
   * - JSON integer: count, is_proportion = false
   * - JSON float: proportion, is_proportion = true
   *
   * @throws std::runtime_error on invalid input.
   */
  VarsSpec parse_vars(const nlohmann::json& j);
}
