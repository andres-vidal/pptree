#include "ConfusionMatrix.hpp"

#include "Stats.hpp"
#include "Invariant.hpp"

#include <set>
#include <stdexcept>

using namespace models::types;

namespace models::stats {
  std::map<int, int> get_labels_map(const ResponseVector& groups) {
    std::set<int> labels_set = unique(groups);

    std::map<int, int> labels_map;
    int i = 0;

    for (int label : labels_set) {
      labels_map[label] = i++;
    }

    return labels_map;
  }

  ConfusionMatrix::ConfusionMatrix(
    const ResponseVector& predictions,
    const ResponseVector& actual)
    : label_index(get_labels_map(actual)) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("cannot compute confusion matrix if predictions and observations have different sizes");
    }

    values = Matrix<int>::Zero(
      static_cast<int>(label_index.size()),
      static_cast<int>(label_index.size()));

    for (int i = 0; i < predictions.rows(); i++) {
      const int actual_index     = label_index.at(actual(i));
      const int prediction_index = label_index.at(predictions(i));

      values(actual_index, prediction_index)++;
    }
  }

  types::Vector<float> ConfusionMatrix::class_errors() const {
    Matrix<int> error_matrix = values;
    error_matrix.diagonal().setZero();

    Vector<int> row_sums   = values.rowwise().sum();
    Vector<int> row_errors = error_matrix.rowwise().sum();

    return row_errors.array().cast<float>() / row_sums.array().cast<float>();
  }

  float ConfusionMatrix::error() const {
    return 1.0f - static_cast<float>(values.trace()) / static_cast<float>(values.sum());
  }
}
