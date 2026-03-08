#pragma once

#include "utils/Types.hpp"
#include "utils/Invariant.hpp"

#include <set>
#include <stdexcept>
#include <pcg_random.hpp>

namespace pptree::stats {
  using RNG = pcg32;


  /**
   * @brief Sort a feature matrix and a response vector by the response values.
   *
   * @param x  Feature matrix.
   * @param y  Response vector.
   */
  void sort(types::FeatureMatrix& x, types::ResponseVector& y);

  /**
   * @brief Unique values of a response vector.
   *
   * @param column  Response vector.
   * @return        Set of unique response values.
   */
  std::set<types::Response> unique(const types::ResponseVector& column);

  /**
   * @brief Accuracy of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual response vector.
   * @return             Accuracy (0 to 1).
   */
  float accuracy(const types::ResponseVector& predictions, const types::ResponseVector& actual);

  /**
   * @brief Error rate of a prediction.
   *
   * @param predictions  Predicted response vector.
   * @param actual       Actual response vector.
   * @return             Error rate (0 to 1).
   */
  double error_rate(const types::ResponseVector& predictions, const types::ResponseVector& actual);

  /**
   * @brief Sample standard deviation of a vector.
   *
   * @param data  Feature vector with at least one row.
   * @return      Sample standard deviation.
   */
  double sd(const types::FeatureVector& data);

  /**
   * @brief Column-wise sample standard deviation of a matrix.
   *
   * @param data  Feature matrix with at least 2 rows.
   * @return      FeatureVector of size p (one σ per column).
   */
  types::FeatureVector sd(const types::FeatureMatrix& data);
}
