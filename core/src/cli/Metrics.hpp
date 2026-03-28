/**
 * @file Metrics.hpp
 * @brief Compute model metrics (confusion matrices, variable importance)
 *        and add them to a JSON model representation.
 */
#pragma once

#include "models/Model.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ppforest2::cli {
  /**
   * @brief Compute and add metrics to a model JSON object.
   *
   * Computes training confusion matrix, OOB error/confusion matrix (for forests),
   * and variable importance, then adds them to @p model_data.
   *
   * @param model_data   JSON object to augment with metrics.
   * @param model        The trained model.
   * @param x            Training feature matrix.
   * @param y            Training response vector.
   * @param group_names  Group label names (empty for integer labels).
   * @param seed         Seed for permutation importance (default: 42).
   */
  void compute_metrics(
    nlohmann::json&                 model_data,
    const Model&                    model,
    const types::FeatureMatrix&     x,
    const types::ResponseVector&    y,
    const std::vector<std::string>& group_names,
    int                             seed = 42);
}
