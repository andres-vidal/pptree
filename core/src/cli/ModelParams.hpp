/**
 * @file ModelParams.hpp
 * @brief Model training parameters shared by train and evaluate.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::cli {
  /** @brief Model training parameters shared by train and evaluate. */
  struct ModelParams {
    int size        = 100;
    float lambda    = 0.5;
    int threads     = -1;
    int seed        = -1;
    float p_vars    = -1;
    int n_vars      = -1;
    int max_retries = 3;
    std::string p_vars_input;

    /** @brief Explicit strategy inputs (--X flags). */
    std::string pp_input;
    std::string vars_input;
    std::string cutpoint_input;
    std::string stop_input;
    std::string binarize_input;
    std::string partition_input;
    std::string leaf_input;

    /** @brief Strategy JSON objects (from CLI strings or config file). */
    nlohmann::json pp_config;
    nlohmann::json vars_config;
    nlohmann::json cutpoint_config;
    nlohmann::json stop_config;
    nlohmann::json binarize_config;
    nlohmann::json partition_config;
    nlohmann::json leaf_config;

    bool used_default_seed    = false;
    bool used_default_threads = false;
    bool used_default_vars    = false;
  };
}
