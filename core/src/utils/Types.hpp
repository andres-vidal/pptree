#pragma once

#include <Eigen/Dense>

/**
 * @brief Core numeric type aliases for the ppforest2 library.
 *
 * All matrix and vector types are Eigen dynamic-size types.  The
 * scalar precision for features is controlled by the compile-time
 * flag PPFOREST2_DOUBLE_PRECISION (float by default).
 */
namespace ppforest2::types {
/** @brief Scalar type for feature values (float or double). */
#ifdef PPFOREST2_DOUBLE_PRECISION
  using Feature = double;
#else
  using Feature = float;
#endif

  /** @brief Scalar type for group labels (integer). */
  using Response = int;

  /** @brief Dynamic-size matrix of feature values. */
  using FeatureMatrix = Eigen::Matrix<Feature, Eigen::Dynamic, Eigen::Dynamic>;
  /** @brief Dynamic-size column vector of feature values. */
  using FeatureVector = Eigen::Matrix<Feature, Eigen::Dynamic, 1>;

  /** @brief Dynamic-size column vector of group labels. */
  using ResponseVector = Eigen::Matrix<Response, Eigen::Dynamic, 1>;

  /** @brief Generic dynamic-size matrix. */
  template<typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /** @brief Generic dynamic-size column vector. */
  template<typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
}
