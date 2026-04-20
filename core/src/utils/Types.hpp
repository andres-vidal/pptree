#pragma once

#include "utils/UserError.hpp"

#include <Eigen/Dense>
#include <string>
#include <string_view>

/**
 * @brief Core numeric type aliases for the ppforest2 library.
 *
 * All matrix and vector types are Eigen dynamic-size types.  Feature
 * precision is single-precision (`float`), which is sufficient for
 * classification.  If a future strategy (e.g. regression) needs higher
 * precision internally, it can cast to `double` within its own scope.
 *
 * GroupId is the internal integer type for group labels, partition keys,
 * and confusion matrices.  Outcome is the prediction type — currently
 * an alias for int, but will become Feature (float) to support regression.
 */
namespace ppforest2::types {
  /** @brief Scalar type for feature values. */
  using Feature = float;

  /** @brief Scalar type for internal group labels (integer). Used as map keys, set elements, and partition indices. */
  using GroupId = int;

  /** @brief Scalar type for predictions (float for both classification and regression). */
  using Outcome = Feature;

  /** @brief Dynamic-size matrix of feature values. */
  using FeatureMatrix = Eigen::Matrix<Feature, Eigen::Dynamic, Eigen::Dynamic>;

  /** @brief Dynamic-size column vector of feature values. */
  using FeatureVector = Eigen::Matrix<Feature, Eigen::Dynamic, 1>;

  /** @brief Dynamic-size column vector of internal group labels. */
  using GroupIdVector = Eigen::Matrix<GroupId, Eigen::Dynamic, 1>;

  /** @brief Dynamic-size column vector of predictions. */
  using OutcomeVector = Eigen::Matrix<Outcome, Eigen::Dynamic, 1>;

  /** @brief Generic dynamic-size matrix. */
  template<typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  /** @brief Generic dynamic-size column vector. */
  template<typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /** @brief Training mode. */
  enum class Mode : uint8_t { Classification, Regression };

  /**
   * @brief Canonical string form of a training mode.
   */
  inline std::string to_string(Mode mode) {
    return mode == Mode::Regression ? "regression" : "classification";
  }

  /**
   * @brief Training mode from string
   */
  inline Mode mode_from_string(std::string_view s) {
    if (s == "classification") {
      return Mode::Classification;
    }

    if (s == "regression") {
      return Mode::Regression;
    }

    throw UserError("Invalid mode '" + std::string(s) + "'. Expected 'classification' or 'regression'.");
  }

}
