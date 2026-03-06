/**
 * @file Presentation.hpp
 * @brief Terminal announcement helpers, model statistics, and confusion
 *        matrix display for the pptree CLI.
 */
#pragma once

#include "cli/Color.hpp"
#include "cli/CLIOptions.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace pptree::cli {
  /**
   * @brief Return a muted "(default)" tag if the value was auto-detected.
   * @param is_default Whether the parameter used its default value.
   * @return A styled " (default)" string, or empty if not default.
   */
  inline std::string default_tag(bool is_default) {
    if (!is_default) return "";

    return " " + muted("(default)");
  }

  /**
   * @brief Print the training configuration summary to stdout.
   *
   * Displays the model type (forest vs single tree), hyperparameters,
   * and optional train/test split sizes. Respects the quiet flag.
   *
   * @param params  The CLI options struct.
   * @param n_train Number of training samples (0 to omit split info).
   * @param n_test  Number of test samples (0 to omit split info).
   */
  inline void announce_configuration(
    const CLIOptions& params,
    int               n_train = 0,
    int               n_test  = 0) {
    if (params.quiet) return;

    if (params.trees > 0) {
      fmt::print("Training {} with {} trees\n", emphasis("random forest"), emphasis(std::to_string(params.trees)));
      fmt::print("-- variables per split: {} ({}% of features){}\n", emphasis(std::to_string(params.n_vars)), params.p_vars * 100, default_tag(params.used_default_vars));
      fmt::print("-- threads: {}{}\n", emphasis(std::to_string(params.threads)), default_tag(params.used_default_threads));
      fmt::print("-- seed: {}{}\n", emphasis(std::to_string(params.seed)), default_tag(params.used_default_seed));
    } else {
      fmt::print("Training {} (using all features)\n", emphasis("single decision tree"));
    }

    fmt::print("-- method: {} (lambda={})\n", emphasis(params.lambda == 0 ? "LDA" : "PDA"), params.lambda);

    if (n_train > 0 && n_test > 0) {
      fmt::print("\nData split into:\n"
        "-- training: {} samples ({}%)\n"
        "-- test:     {} samples ({}%)\n",
        emphasis(std::to_string(n_train)), params.train_ratio * 100,
        emphasis(std::to_string(n_test)), (1 - params.train_ratio) * 100);
    }

    fmt::print("\n");
  }

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

    double mean_time() const {
      return tr_times.mean();
    }

    double mean_tr_error() const {
      return tr_error.mean();
    }

    double mean_te_error() const {
      return te_error.mean();
    }

    double std_time() const {
      return stats::sd(tr_times);
    }

    double std_tr_error() const {
      return stats::sd(tr_error);
    }

    double std_te_error() const {
      return stats::sd(te_error);
    }

    /** @brief Serialize to JSON including per-iteration breakdown. */
    nlohmann::json to_json() const {
      nlohmann::json j = {
        { "runs",             tr_times.size() },
        { "mean_time_ms",     mean_time() },
        { "std_time_ms",      std_time() },
        { "mean_train_error", mean_tr_error() },
        { "std_train_error",  std_tr_error() },
        { "mean_test_error",  mean_te_error() },
        { "std_test_error",   std_te_error() }
      };

      if (peak_rss_bytes >= 0) {
        j["peak_rss_bytes"] = peak_rss_bytes;
        j["peak_rss_mb"]    = static_cast<double>(peak_rss_bytes) / (1024.0 * 1024.0);
      }

      // Per-iteration data
      nlohmann::json iterations = nlohmann::json::array();
      for (int i = 0; i < tr_times.size(); ++i) {
        iterations.push_back({
          { "train_time_ms", tr_times[i] },
          { "train_error",   tr_error[i] },
          { "test_error",    te_error[i] }
        });
      }

      j["iterations"] = iterations;

      return j;
    }
  };

  /**
   * @brief Print evaluation results (timing, errors, memory) to stdout.
   * @param stats The aggregated model statistics.
   */
  inline void announce_results(const ModelStats& stats) {
    fmt::print("{} ({} runs):\n"
      "-- training time: {:.2f}ms ± {:.2f}ms\n"
      "-- train error:   {:.2f}%  ± {:.2f}%\n"
      "-- test error:    {:.2f}%  ± {:.2f}%\n",
      emphasis("Evaluation results"), stats.tr_times.size(),
      stats.mean_time(), stats.std_time(),
      stats.mean_tr_error() * 100, stats.std_tr_error() * 100,
      stats.mean_te_error() * 100, stats.std_te_error() * 100);

    if (stats.peak_rss_bytes >= 0) {
      double mb = static_cast<double>(stats.peak_rss_bytes) / (1024.0 * 1024.0);
      fmt::print("-- peak RSS:      {:.1f} MB\n", mb);
    }
  }

  /**
   * @brief Print a ranked variable importance table to stdout.
   *
   * Columns: rank, variable name, σ (scale), VI2 (projections), VI3 (weighted),
   * VI1 (permuted). Rows are sorted by VI2 descending. The VI1 column is
   * omitted when its vector is empty. At most max_rows rows are printed.
   *
   * @param vi1       Permuted importance vector (size p, or size-0 to omit).
   * @param vi2       Projections importance vector (size p).
   * @param vi3       Weighted projections importance vector (size p).
   * @param scale     Per-variable standard deviation used to scale VI2/VI3.
   * @param max_rows  Maximum number of rows to print (0 = all).
   */
  inline void print_variable_importance(
    const types::FeatureVector& vi1,
    const types::FeatureVector& vi2,
    const types::FeatureVector& vi3,
    const types::FeatureVector& scale,
    int                         max_rows = 20) {
    const int p = static_cast<int>(vi2.size());

    std::vector<int> order(static_cast<std::size_t>(p));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&vi2](int a, int b) {
      return vi2(a) > vi2(b);
    });

    const bool show_vi1 = vi1.size() == vi2.size();
    const bool show_vi3 = vi3.size() == vi2.size();
    const int rows      = (max_rows > 0 && p > max_rows) ? max_rows : p;

    fmt::print("{}\n\n", emphasis("Variable Importance:"));

    std::string sigma_hdr = muted(fmt::format("{:>10}", "σ"));

    if (show_vi3 && show_vi1) {
      fmt::print("{:>5}  {:<10}  {}  {:>12}  {:>12}  {:>12}\n", "Rank", "Variable", sigma_hdr, "Projection", "Weighted", "Permuted");
      fmt::print("{:>5}  {:<10}  {:>10}  {:>12}  {:>12}  {:>12}\n", "----", "--------", "", "(VI2)", "(VI3)", "(VI1)");
    } else if (show_vi3) {
      fmt::print("{:>5}  {:<10}  {}  {:>12}  {:>12}\n",  "Rank", "Variable", sigma_hdr, "Projection", "Weighted");
      fmt::print("{:>5}  {:<10}  {:>10}  {:>12}  {:>12}\n",  "----", "--------", "", "(VI2)", "(VI3)");
    } else {
      fmt::print("{:>5}  {:<10}  {}  {:>12}\n",  "Rank", "Variable", sigma_hdr, "Projection");
      fmt::print("{:>5}  {:<10}  {:>10}  {:>12}\n", "----", "--------", "", "(VI2)");
    }

    for (int rank = 0; rank < rows; ++rank) {
      int j         = order[static_cast<std::size_t>(rank)];
      std::string v = fmt::format("x{}", j + 1);

      if (show_vi3 && show_vi1) {
        fmt::print("{:>5}  {:<10}  {:>10.4f}  {:>12.6f}  {:>12.6f}  {:>12.6f}\n", rank + 1, v, scale(j), vi2(j), vi3(j), vi1(j));
      } else if (show_vi3) {
        fmt::print("{:>5}  {:<10}  {:>10.4f}  {:>12.6f}  {:>12.6f}\n",  rank + 1, v, scale(j), vi2(j), vi3(j));
      } else {
        fmt::print("{:>5}  {:<10}  {:>10.4f}  {:>12.6f}\n", rank + 1, v, scale(j), vi2(j));
      }
    }

    if (rows < p) {
      fmt::print("{}\n", muted(fmt::format("  ... {} more variables not shown", p - rows)));
    }

    fmt::print("\n");
  }

  /**
   * @brief Build a JSON object from VI vectors, suitable for embedding in the
   *        saved model file.
   *
   * @param vi1    Permuted importance (size p, or size-0 to omit).
   * @param vi2    Projections importance (size p).
   * @param vi3    Weighted projections importance (size p).
   * @param scale  Per-variable standard deviation used to scale VI2/VI3.
   * @return JSON object with "scale", "projections", "weighted_projections",
   *         and optionally "permuted" arrays (indexed by variable, 0-based).
   */
  inline nlohmann::json vi_to_json(
    const types::FeatureVector& vi1,
    const types::FeatureVector& vi2,
    const types::FeatureVector& vi3,
    const types::FeatureVector& scale) {
    const int p = static_cast<int>(vi2.size());

    std::vector<float> scale_vec(scale.data(), scale.data() + p);
    std::vector<float> vi2_vec(vi2.data(), vi2.data() + p);

    nlohmann::json j;
    j["scale"]       = scale_vec;
    j["projections"] = vi2_vec;

    if (vi3.size() == vi2.size()) {
      std::vector<float> vi3_vec(vi3.data(), vi3.data() + p);
      j["weighted_projections"] = vi3_vec;
    }

    if (vi1.size() == vi2.size()) {
      std::vector<float> vi1_vec(vi1.data(), vi1.data() + p);
      j["permuted"] = vi1_vec;
    }

    return j;
  }

  /**
   * @brief Print a formatted confusion matrix to stdout.
   *
   * Displays the confusion matrix with class labels, diagonal highlighting,
   * and per-class error rates.
   *
   * @param cm The confusion matrix to print.
   */
  inline void print_confusion_matrix(const stats::ConfusionMatrix& cm) {
    int n          = cm.values.rows();
    auto class_err = cm.class_errors();

    fmt::print("{}\n\n", emphasis("Confusion Matrix:"));

    // Header row: class labels
    fmt::print("    ");
    for (const auto& [label, idx] : cm.label_index) {
      fmt::print("{:>5}", label);
    }

    fmt::print("  Error\n");

    // Data rows
    for (const auto& [label, row_idx] : cm.label_index) {
      fmt::print("{:>4}", label);
      for (const auto& [col_label, col_idx] : cm.label_index) {
        int val          = cm.values(row_idx, col_idx);
        std::string cell = fmt::format("{:>4}", val);

        if (row_idx == col_idx) {
          fmt::print(" {}", success(cell));
        } else if (val > 0) {
          fmt::print(" {}", error(cell));
        } else {
          fmt::print(" {}", muted(cell));
        }
      }

      fmt::print("  {:.1f}%\n", class_err[row_idx] * 100);
    }

    fmt::print("\n");
  }
}
