#include "stats/RegressionMetrics.hpp"

#include "utils/Invariant.hpp"

#include <cmath>
#include <stdexcept>

using namespace ppforest2::types;

namespace ppforest2::stats {
  static void validate_inputs(OutcomeVector const& predictions, OutcomeVector const& actual) {
    if (predictions.size() != actual.size()) {
      throw std::invalid_argument("Predictions and actual vectors must have the same size.");
    }

    if (predictions.size() == 0) {
      throw std::invalid_argument("Vectors must not be empty.");
    }
  }

  double mse(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);

    double sum = 0.0;

    for (int i = 0; i < predictions.size(); ++i) {
      double diff = static_cast<double>(predictions(i)) - static_cast<double>(actual(i));
      sum += diff * diff;
    }

    return sum / static_cast<double>(predictions.size());
  }

  double mae(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);

    double sum = 0.0;

    for (int i = 0; i < predictions.size(); ++i) {
      double diff = static_cast<double>(predictions(i)) - static_cast<double>(actual(i));
      sum += std::abs(diff);
    }

    return sum / static_cast<double>(predictions.size());
  }

  double r_squared(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);

    double mean_actual = 0.0;

    for (int i = 0; i < actual.size(); ++i) {
      mean_actual += static_cast<double>(actual(i));
    }

    mean_actual /= static_cast<double>(actual.size());

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (int i = 0; i < predictions.size(); ++i) {
      double diff_res = static_cast<double>(predictions(i)) - static_cast<double>(actual(i));
      double diff_tot = static_cast<double>(actual(i)) - mean_actual;
      ss_res += diff_res * diff_res;
      ss_tot += diff_tot * diff_tot;
    }

    if (ss_tot == 0.0) {
      return 0.0;
    }

    return 1.0 - ss_res / ss_tot;
  }

  RegressionMetrics::RegressionMetrics(OutcomeVector const& predictions, OutcomeVector const& actual)
      : mse(stats::mse(predictions, actual))
      , mae(stats::mae(predictions, actual))
      , r_squared(stats::r_squared(predictions, actual)) {}
}
