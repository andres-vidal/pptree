/**
 * @file Validation.hpp
 * @brief Central config validation for all CLI subcommands.
 *
 * Every subcommand (train, evaluate, benchmark) produces a JSON config
 * that is validated centrally before being applied to Params.
 * Central defaults exist for shared parameters; benchmark scenarios
 * skip them (scenarios must be explicit).
 */
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ppforest2::cli {
  /**
   * @brief Collects validation errors grouped by label.
   *
   * Use begin() to start a new group (e.g., a scenario name),
   * then pass errors() to validation functions that append to
   * a std::vector<std::string>&.
   */
  /**
   * @brief Append an error message if condition is false.
   */
  inline void check(bool condition, std::string const& message, std::vector<std::string>& errors) {
    if (!condition) {
      errors.push_back(message);
    }
  }

  /**
   * @brief Default values for shared training parameters.
   *
   * Used by train and evaluate to fill in unset fields before validation.
   * Benchmark does not apply these — its scenarios must be explicit.
   */
  nlohmann::json training_defaults();

  /**
   * @brief Validate a training config JSON.
   *
   * Checks all required fields, ranges, and mutual exclusions.
   * Appends error messages to @p errors (empty if valid).
   *
   * @param config      The config to validate.
   * @param errors      Output vector to accumulate error messages.
   */
  void validate_training_config(nlohmann::json const& config, std::vector<std::string>& errors);

  struct Params;

  /**
   * @brief Validate training-related params and throw on errors.
   *
   * Builds a config from params, validates it, and throws UserError
   * with all collected messages if any checks fail.
   */
  void validate_params(Params const& params);
}
