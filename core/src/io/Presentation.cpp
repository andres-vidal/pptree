/**
 * @file Presentation.cpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#include "io/Presentation.hpp"

#include <fmt/format.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace pptree::cli {
  nlohmann::json ModelStats::to_json() const {
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

  void announce_results(const ModelStats& stats) {
    using namespace pptree::io;

    fmt::print("{} ({} runs):\n"
      "-- training time: {:.2f}ms \u00b1 {:.2f}ms\n"
      "-- train error:   {:.2f}%  \u00b1 {:.2f}%\n"
      "-- test error:    {:.2f}%  \u00b1 {:.2f}%\n",
      emphasis("Evaluation results"), stats.tr_times.size(),
      stats.mean_time(), stats.std_time(),
      stats.mean_tr_error() * 100, stats.std_tr_error() * 100,
      stats.mean_te_error() * 100, stats.std_te_error() * 100);

    if (stats.peak_rss_bytes >= 0) {
      double mb = static_cast<double>(stats.peak_rss_bytes) / (1024.0 * 1024.0);
      fmt::print("-- peak RSS:      {:.1f} MB\n", mb);
    }
  }

  void print_variable_importance(
    const types::FeatureVector& vi1,
    const types::FeatureVector& vi2,
    const types::FeatureVector& vi3,
    const types::FeatureVector& scale,
    int                         max_rows) {
    using namespace pptree::io;

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

    std::string sigma_hdr = muted(fmt::format("{:>10}", "\xcf\x83"));

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

  nlohmann::json vi_to_json(
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

  void print_confusion_matrix(const stats::ConfusionMatrix& cm) {
    using namespace pptree::io;

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
