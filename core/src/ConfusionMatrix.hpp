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
  };
}
