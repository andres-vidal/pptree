#pragma once

#include "Types.hpp"

#include <map>

namespace models::stats {
  std::map<int, int> get_labels_map(const types::ResponseVector& groups);

  struct ConfusionMatrix {
    types::Matrix<int> values;
    std::map<int, int> label_index;

    ConfusionMatrix(const types::ResponseVector& predictions, const types::ResponseVector& actual);

    types::Vector<float> class_errors() const;
    float error() const;
  };
}
