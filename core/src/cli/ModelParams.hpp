/**
 * @file ModelParams.hpp
 * @brief Model training parameters shared by train and evaluate.
 */
#pragma once

#include <string>

namespace ppforest2::cli {
  /** @brief Model training parameters shared by train and evaluate. */
  struct ModelParams {
    int trees       = 100;
    float lambda    = 0.5;
    int threads     = -1;
    int seed        = -1;
    float p_vars    = -1;
    int n_vars      = -1;
    int max_retries = 3;
    std::string vars_input;

    bool used_default_seed    = false;
    bool used_default_threads = false;
    bool used_default_vars    = false;
  };
}
