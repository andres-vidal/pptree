#pragma once

#include <Eigen/Dense>

/**
 * @brief Core numeric type aliases for the ppforest2 library.
 *
 * All matrix and vector types are Eigen dynamic-size types.  Feature
 * precision is single-precision (`float`), which is sufficient for
 * classification.  If a future strategy (e.g. regression) needs higher
 * precision internally, it can cast to `double` within its own scope.
 */
namespace ppforest2::types {
  /** @brief Scalar type for feature values. */
  using Feature = float;

  /** @brief Scalar type for group labels (integer). */
  using Outcome = int;

  /** @brief Dynamic-size matrix of feature values. */
  using FeatureMatrix = Eigen::Matrix<Feature, Eigen::Dynamic, Eigen::Dynamic>;

  /** @brief Dynamic-size column vector of feature values. */
  using FeatureVector = Eigen::Matrix<Feature, Eigen::Dynamic, 1>;

  /** @brief Dynamic-size column vector of group labels. */
  using OutcomeVector = Eigen::Matrix<Outcome, Eigen::Dynamic, 1>;

  /** @brief Generic dynamic-size matrix. */
  template<typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /** @brief Generic dynamic-size column vector. */
  template<typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
}
