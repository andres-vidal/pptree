#pragma once

#include "utils/Types.hpp"

namespace ppforest2::stats {
  /**
   * @brief Regression evaluation metrics.
   *
   * Computes MSE, MAE, and R-squared from predictions and actual values.
   *
   * @code
   *   RegressionMetrics metrics(predictions, actual);
   *   double mse = metrics.mse;
   *   double r2  = metrics.r_squared;
   * @endcode
   */
  struct RegressionMetrics {
    double mse       = 0.0; ///< Mean squared error.
    double mae       = 0.0; ///< Mean absolute error.
    double r_squared = 0.0; ///< Coefficient of determination (R²).

    /** @brief Default-construct empty metrics. */
    RegressionMetrics() = default;

    /**
     * @brief Compute metrics from predictions and actual values.
     *
     * @param predictions  Predicted response vector.
     * @param actual       True response vector (same size).
     * @throws std::invalid_argument If sizes differ or vectors are empty.
     */
    RegressionMetrics(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);
  };

  /**
   * @brief Compute MSE between predictions and actual values.
   */
  double mse(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /**
   * @brief Compute MAE between predictions and actual values.
   */
  double mae(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /**
   * @brief Compute R-squared between predictions and actual values.
   */
  double r_squared(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);
}
