#include "stats/Stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

using namespace pptree::types;

namespace pptree::stats {
  FeatureMatrix standardize(const FeatureMatrix& data) {
    invariant(data.rows() >= 2, "standardize requires at least 2 rows");

    FeatureMatrix centered = data.rowwise() - data.colwise().mean();
    Vector<Feature> sd     =
      (centered.array().square().colwise().sum() / (data.rows() - 1)).sqrt();

    return centered.array().rowwise() / sd.transpose().array();
  }

  void sort(FeatureMatrix& x, ResponseVector& y) {
    std::vector<int> indices(x.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(),
      [&y](int idx1, int idx2) {
        return y(idx1) < y(idx2);
      });

    x = x(indices, Eigen::all).eval();
    y = y(indices, Eigen::all).eval();
  }

  std::set<Response> unique(const ResponseVector& column) {
    std::set<Response> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  float accuracy(const ResponseVector& predictions, const ResponseVector& actual) {
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

  double error_rate(const ResponseVector& predictions, const ResponseVector& actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    return 1.0 - accuracy(predictions, actual);
  }

  double sd(const FeatureVector& data) {
    if (data.rows() == 0) {
      throw std::invalid_argument("sd: data must have at least one row");
    }

    if (data.rows() == 1) {
      return 0.0;
    }

    return std::sqrt((data.array() - data.mean()).square().sum() / (data.rows() - 1));
  }
}
