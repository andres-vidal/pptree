#pragma once

#include "utils/Types.hpp"
#include "utils/Invariant.hpp"

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
  void sort(types::FeatureMatrix& x, types::OutcomeVector& y);

  /**
   * @brief Unique values of a response vector.
   *
   * @param column  Outcome vector.
   * @return        Set of unique response values.
   */
  std::set<types::Outcome> unique(types::OutcomeVector const& column);

  /**
   * @brief Accuracy of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual response vector.
   * @return             Accuracy (0 to 1).
   */
  float accuracy(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /**
   * @brief Error rate of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual response vector.
   * @return             Error rate (0 to 1).
   */
  double error_rate(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /**
   * @brief Sample standard deviation of a vector.
   *
   * @param data  Feature vector with at least one row.
   * @return      Sample standard deviation.
   */
  double sd(types::FeatureVector const& data);

  /**
   * @brief Column-wise sample standard deviation of a matrix.
   *
   * @param data  Feature matrix with at least 2 rows.
   * @return      FeatureVector of size p (one σ per column).
   */
  types::FeatureVector sd(types::FeatureMatrix const& data);
}
