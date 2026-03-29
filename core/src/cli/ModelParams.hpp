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
    std::string vars_input;

    /** @brief Explicit strategy inputs (--pp, --dr, --sr flags). */
    std::string pp_input;
    std::string dr_input;
    std::string sr_input;

    /** @brief Strategy JSON objects loaded from config file (pp/dr/sr). */
    nlohmann::json pp_config;
    nlohmann::json dr_config;
    nlohmann::json sr_config;

    bool used_default_seed    = false;
    bool used_default_threads = false;
    bool used_default_vars    = false;
  };
}
