#pragma once

#include "Data.hpp"

#include <map>

namespace models::stats {
  inline std::map<int, int> get_labels_map(const DataColumn<int> &groups) {
    std::set<int> labels_set = unique(groups);
    std::map<int, int> labels_map;

    int i = 0;

    for (int label : labels_set) {
      labels_map[label] = i++;
    }

    return labels_map;
  }

  struct ConfusionMatrix {
    Data<int> values;
    std::map<int, int> label_index;

    ConfusionMatrix(const DataColumn<int> &predictions, const DataColumn<int> &actual) : label_index(get_labels_map(actual)) {
      if (predictions.rows() != actual.rows()) {
        throw std::invalid_argument("cannot compute confusion matrix if predictions and observations have different sizes");
      }

      this->values = Data<int>::Zero(label_index.size(), label_index.size());

      for (int i = 0; i < predictions.rows(); i++) {
        const int actual_index = label_index[actual(i)];
        const int prediction_index = label_index[predictions(i)];

        this->values(actual_index, prediction_index)++;
      }
    }

    DataColumn<long double> class_errors() const {
      Data<int> error_matrix = values;
      error_matrix.diagonal().setZero();

      DataColumn<int> row_sums = values.rowwise().sum();
      DataColumn<int> row_errors = error_matrix.rowwise().sum();

      return row_errors.array().cast<long double>() / row_sums.array().cast<long double>();
    }

    long double error() const {
      return 1 - math::trace(values) / (double)math::sum(values);
    }
  };
}
