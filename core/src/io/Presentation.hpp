/**
 * @file Presentation.hpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#pragma once

#include "io/Color.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>

namespace pptree::cli {
  /**
   * @brief Aggregated statistics across multiple training iterations.
   *
   * Stores per-iteration timing and error data, plus the process-wide
   * peak RSS. Provides mean/std accessors and JSON serialization
   * (including a per-iteration breakdown).
   */
  struct ModelStats {
    types::Vector<float> tr_times;
    types::Vector<float> tr_error;
    types::Vector<float> te_error;
    long peak_rss_bytes = -1;

    double mean_time() const { return tr_times.mean(); }
    double mean_tr_error() const { return tr_error.mean(); }
    double mean_te_error() const { return te_error.mean(); }

    double std_time() const { return stats::sd(tr_times); }
    double std_tr_error() const { return stats::sd(tr_error); }
    double std_te_error() const { return stats::sd(te_error); }

    /** @brief Serialize to JSON including per-iteration breakdown. */
    nlohmann::json to_json() const;
  };

  /**
   * @brief Print evaluation results (timing, errors, memory) to stdout.
   * @param stats The aggregated model statistics.
   */
  void print_results(const ModelStats& stats);

  /**
   * @brief Print a ranked variable importance table to stdout.
   *
   * Columns: rank, variable name, sigma (scale), VI2 (projections), VI3 (weighted),
   * VI1 (permuted). Rows are sorted by VI2 descending. The VI1 column is
   * omitted when its vector is empty. At most max_rows rows are printed.
   *
   * @param vi        Variable importance results.
   * @param max_rows  Maximum number of rows to print (0 = all).
   */
  void print_variable_importance(
    const VariableImportance& vi,
    int                       max_rows = 20);

  /**
   * @brief Print a formatted confusion matrix to stdout.
   *
   * Displays the confusion matrix with class labels, diagonal highlighting,
   * and per-class error rates.
   *
   * @param cm The confusion matrix to print.
   */
  void print_confusion_matrix(const stats::ConfusionMatrix& cm);
}
