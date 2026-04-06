#include "stats/Stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

using namespace ppforest2::types;

namespace ppforest2::stats {
  void sort(FeatureMatrix& x, OutcomeVector& y) {
    std::vector<int> indices(x.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [&y](int idx1, int idx2) { return y(idx1) < y(idx2); });

    x = x(indices, Eigen::all).eval();
    y = y(indices, Eigen::all).eval();
  }

  std::set<Outcome> unique(OutcomeVector const& column) {
    std::set<Outcome> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  float accuracy(OutcomeVector const& predictions, OutcomeVector const& actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    int correct = 0;
    for (int i = 0; i < predictions.rows(); i++) {
      if (predictions(i) == actual(i)) {
        correct++;
      }
    }

    return static_cast<float>(correct) / static_cast<float>(predictions.rows());
  }

  double error_rate(OutcomeVector const& predictions, OutcomeVector const& actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    return 1.0 - accuracy(predictions, actual);
  }

  double sd(FeatureVector const& data) {
    if (data.rows() == 0) {
      throw std::invalid_argument("sd: data must have at least one row");
    }

    if (data.rows() == 1) {
      return 0.0;
    }

    return std::sqrt((data.array() - data.mean()).square().sum() / (data.rows() - 1));
  }

  FeatureVector sd(FeatureMatrix const& data) {
    invariant(data.rows() >= 2, "sd: matrix must have at least 2 rows");

    FeatureMatrix centered = data.rowwise() - data.colwise().mean();

    return (centered.array().square().colwise().sum() / static_cast<Feature>(data.rows() - 1)).sqrt();
  }
}
