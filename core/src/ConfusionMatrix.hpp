/**
 * @file ConfusionMatrix.hpp
 * @brief Confusion matrix for classification model evaluation.
 */
#pragma once

#include "Types.hpp"

#include <map>

#include <nlohmann/json.hpp>

namespace models::stats {
  /**
   * @brief Build a sorted mapping from unique class labels to contiguous indices.
   * @param groups A response vector containing class labels.
   * @return A map from label value to its 0-based index.
   */
  std::map<int, int> get_labels_map(const types::ResponseVector& groups);

  /**
   * @brief A confusion matrix comparing predicted vs actual class labels.
   *
   * Rows correspond to actual classes and columns to predicted classes.
   * Provides overall error rate, per-class error rates, JSON serialization,
   * and formatted terminal printing with diagonal highlighting and marginal errors.
   */
  struct ConfusionMatrix {
    types::Matrix<int> values;         ///< The NxN confusion matrix (actual x predicted).
    std::map<int, int> label_index;    ///< Map from class label to matrix row/column index.

    /**
     * @brief Construct a confusion matrix from predictions and actual labels.
     * @param predictions The predicted class labels.
     * @param actual      The true class labels (must have the same size).
     * @throws std::invalid_argument If predictions and actual have different sizes.
     */
    ConfusionMatrix(const types::ResponseVector& predictions, const types::ResponseVector& actual);

    /**
     * @brief Compute per-class error rates.
     * @return A vector of error rates (one per class), where 0 = perfect, 1 = all wrong.
     */
    types::Vector<float> class_errors() const;

    /**
     * @brief Compute the overall error rate (1 - accuracy).
     * @return The proportion of misclassified samples.
     */
    float error() const;

    /**
     * @brief Serialize to JSON with matrix, labels, and class_errors.
     * @return A JSON object containing "matrix", "labels", and "class_errors".
     */
    nlohmann::json to_json() const;

    /**
     * @brief Print the confusion matrix to stdout with colored diagonal
     *        and per-row marginal error percentages.
     */
    void print() const;
  };
}
