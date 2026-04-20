#pragma once

#include "utils/Types.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <pcg_random.hpp>

/**
 * @brief Statistical infrastructure for training and evaluation.
 *
 * Provides the random number generator (pcg32), discrete uniform
 * sampling (Lemire's method), grouped-observation bookkeeping
 * (GroupPartition), confusion matrices, data simulation, and basic
 * descriptive statistics used throughout the training pipeline.
 */
namespace ppforest2::stats {
  using RNG = pcg32;


  /**
   * @brief Sort a feature matrix and a response vector by the response values.
   *
   * @param x  Feature matrix.
   * @param y  Outcome vector.
   */
  void sort(types::FeatureMatrix& x, types::GroupIdVector& y);

  /**
   * @brief Unique group labels in a response vector.
   *
   * @param column  Group ID vector.
   * @return        Set of unique group labels.
   */
  std::set<types::GroupId> unique(types::GroupIdVector const& column);

  /**
   * @brief Accuracy of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual response vector.
   * @return             Accuracy (0 to 1).
   */
  float accuracy(types::OutcomeVector const& predictions, types::GroupIdVector const& actual);

  /**
   * @brief Error rate of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual group label vector.
   * @return             Error rate (0 to 1).
   */
  double error_rate(types::OutcomeVector const& predictions, types::GroupIdVector const& actual);

  /**
   * @brief Convenience overload: accept float-typed class labels.
   *
   * Casts `actual` to `GroupIdVector` locally. Used by the unified training
   * pipeline where `y` is carried as `OutcomeVector` for both modes.
   */
  inline double error_rate(types::OutcomeVector const& predictions, types::OutcomeVector const& actual) {
    types::GroupIdVector const actual_int = actual.cast<types::GroupId>();
    return error_rate(predictions, actual_int);
  }

  /**
   * @brief Sample standard deviation of a vector.
   *
   * @param data  Vector with at least one row.
   * @return      Sample standard deviation.
   */
  template<typename Derived> double sd(Eigen::MatrixBase<Derived> const& data) {
    static_assert(
        Derived::ColsAtCompileTime == 1 || Derived::ColsAtCompileTime == Eigen::Dynamic,
        "sd: expected a vector (single column)"
    );

    if (data.rows() == 0) {
      throw std::invalid_argument("sd: data must have at least one row");
    }

    if (data.rows() == 1) {
      return 0.0;
    }

    double mean = static_cast<double>(data.mean());

    return std::sqrt((data.array().template cast<double>() - mean).square().sum() / (data.rows() - 1));
  }

  /**
   * @brief Column-wise sample standard deviation of a matrix.
   *
   * @param data  Feature matrix with at least 2 rows.
   * @return      FeatureVector of size p (one σ per column).
   */
  types::FeatureVector sd(types::FeatureMatrix const& data);
}
