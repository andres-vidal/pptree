/**
 * @file ConfusionMatrix.cpp
 * @brief Implementation of confusion matrix construction, error computation,
 *        JSON serialization, and formatted terminal output.
 */
#include "ConfusionMatrix.hpp"

#include "Stats.hpp"
#include "Invariant.hpp"
#include "Color.hpp"

#include <set>
#include <stdexcept>
#include <sstream>
#include <fmt/format.h>

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

  nlohmann::json ConfusionMatrix::to_json() const {
    nlohmann::json j;

    std::vector<std::vector<int> > matrix_data;
    for (int i = 0; i < values.rows(); ++i) {
      std::vector<int> row;
      for (int col = 0; col < values.cols(); ++col) {
        row.push_back(values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<int> labels;
    for (const auto& [label, idx] : label_index) {
      labels.push_back(label);
    }

    j["labels"] = labels;

    auto ce = class_errors();
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
    j["class_errors"] = ce_vec;

    return j;
  }

  void ConfusionMatrix::print() const {
    const int cell_width = 6;
    auto ce              = class_errors();

    fmt::print("Confusion Matrix:\n");

    fmt::print("     ");
    for (const auto& [label, idx] : label_index) {
      fmt::print("{:>{}}", label, cell_width);
    }

    fmt::print("{:>{}}\n", "Error", cell_width + 2);

    int row_idx = 0;
    for (const auto& [label, idx] : label_index) {
      fmt::print("{:>4} ", label);
      for (int col = 0; col < values.cols(); ++col) {
        std::string cell  = std::to_string(values(idx, col));
        const int pad_len = std::max(0, cell_width - static_cast<int>(cell.size()));
        std::string padded(static_cast<size_t>(pad_len), ' ');
        padded += cell;

        if (col == idx) {
          fmt::print("{}", pptree::success(pptree::emphasis(padded)));
        } else {
          fmt::print("{}", padded);
        }
      }

      fmt::print("{:>{}}\n", fmt::format("{:.1f}%", ce[row_idx] * 100), cell_width + 2);
      row_idx++;
    }
  }
}
