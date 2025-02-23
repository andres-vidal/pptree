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
  DataColumn<T> select_rows(
    const DataColumn<T> &   data,
    const std::vector<int> &indices) {
    DataColumn<T> result(indices.size());

    for (std::size_t i = 0; i < indices.size(); i++) {
      result(i) = data(indices[i]);
    }

    return result;
  }

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> & data,
    const std::set<int> & indices) {
    return select_rows(data, std::vector<int>(indices.begin(), indices.end()));
  }

  template<typename T>
  T sd(const DataColumn<T> &data) {
    return sqrt((math::inner_square((data.array() - data.mean()).matrix())).value() / (data.rows() - 1));
  }

  template<typename T>
  DataColumn<T> descale(const DataColumn<T> &data) {
    T scaling_factor = sd(data);

    if (scaling_factor == 0) {
      scaling_factor = 1;
    }

    return data.array() / scaling_factor;
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
  float error_rate(const DataColumn<T> &predictions, const DataColumn<T> &actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    return 1 - accuracy(predictions, actual);
  }
}
