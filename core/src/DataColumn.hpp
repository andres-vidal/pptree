#pragma once

#include "DMatrix.hpp"

#include <set>

namespace models::stats {
  template<typename T>
  using DataColumn = math::DVector<T>;

  template<typename N>
  std::set<N> unique(const DataColumn<N> &column) {
    std::set<N> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  template<typename T>
  float accuracy(const DataColumn<T> &predictions, const DataColumn<T> &actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    int correct = 0;

    for (int i = 0; i < predictions.rows(); i++) {
      if (predictions(i) == actual(i)) {
        correct++;
      }
    }

    return (float)correct / (float)predictions.rows();
  }

  template<typename T>
  double error_rate(const DataColumn<T> &predictions, const DataColumn<T> &actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    return 1 - accuracy(predictions, actual);
  }

  template<typename T>
  double sd(const DataColumn<T> &data) {
    if (data.rows() == 0) {
      throw std::invalid_argument("sd: data must have at least one row");
    }

    if (data.rows() == 1) {
      return 0;
    }

    return std::sqrt((data.array() - data.mean()).square().sum() / (data.rows() - 1));
  }
}
