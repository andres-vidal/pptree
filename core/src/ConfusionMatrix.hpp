#include "Data.hpp"

namespace models::stats {
  struct ConfusionMatrix {
    Data<int> values;
    std::set<int> labels;

    ConfusionMatrix(DataColumn<int> predictions, DataColumn<int> actual) : labels(unique(actual)) {
      if (predictions.rows() != actual.rows()) {
        throw std::invalid_argument("cannot compute confusion matrix if predictions and observations have different sizes");
      }

      this->values = Data<int>::Zero(labels.size(), labels.size());

      for (int i = 0; i < predictions.rows(); i++) {
        this->values(actual(i), predictions(i))++;
      }
    }

    DataColumn<double> class_errors() {
      Data<int> error_matrix = values;
      error_matrix.diagonal().setZero();

      DataColumn<int> row_sums = values.rowwise().sum();
      DataColumn<int> row_errors = error_matrix.rowwise().sum();

      return row_errors.array().cast<double>() / row_sums.array().cast<double>();
    }

    double error() {
      return 1 - math::trace(values) / (double)math::sum(values);
    }
  };
}
