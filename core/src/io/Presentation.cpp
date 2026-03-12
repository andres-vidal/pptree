/**
 * @file Presentation.cpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#include "io/Presentation.hpp"
#include "io/Table.hpp"

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

  void print_results(pptree::io::Output& out, const ModelStats& stats) {
    using namespace pptree::io;

    out.println("{}", emphasis("Evaluation results"));
    out.newline();

    std::vector<Column> columns = {
      { "Runs",       5,  Align::left },
      { "Time (ms)", 18,  Align::right },
      { "Train Err", 10,  Align::right },
      { "Test Err",  10,  Align::right },
    };

    if (stats.peak_rss_bytes >= 0) {
      columns.push_back({ "Peak RSS", 10, Align::right });
    }

    Row header = header_labels(columns);

    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    std::string time_str = fmt::format("{:.2f} +/- {:.2f}", stats.mean_time(), stats.std_time());
    std::string tr_err   = fmt::format("{:.2f}%", stats.mean_tr_error() * 100);
    std::string te_err   = fmt::format("{:.2f}%", stats.mean_te_error() * 100);

    Row cells = {
      fmt::format("{}", stats.tr_times.size()),
      time_str,
      tr_err,
      te_err,
    };

    if (stats.peak_rss_bytes >= 0) {
      double mb = static_cast<double>(stats.peak_rss_bytes) / (1024.0 * 1024.0);
      cells.push_back(fmt::format("{:.1f} MB", mb));
    }

    out.println("{}", format_row(columns, cells));
    out.newline();
  }

  void print_variable_importance(
    pptree::io::Output&       out,
    const VariableImportance& vi,
    int                       max_rows) {
    using namespace pptree::io;

    const auto& vi1   = vi.permuted;
    const auto& vi2   = vi.projections;
    const auto& vi3   = vi.weighted_projections;
    const auto& scale = vi.scale;

    const int p = static_cast<int>(vi2.size());

    std::vector<int> order(static_cast<std::size_t>(p));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&vi2](int a, int b) {
        return vi2(a) > vi2(b);
      });

    const bool show_vi1 = vi1.size() == vi2.size();
    const bool show_vi3 = vi3.size() == vi2.size();
    const int rows      = (max_rows > 0 && p > max_rows) ? max_rows : p;

    out.println("{}", emphasis("Variable Importance:"));
    out.newline();

    // Build columns conditionally
    std::vector<Column> columns = {
      { "Rank",       5,  Align::left },
      { "Variable",  10,  Align::left },
      { "",          10,  Align::right },
      { "Projection", 12, Align::right },
    };

    if (show_vi3) columns.push_back({ "Weighted", 12, Align::right });

    if (show_vi1) columns.push_back({ "Permuted", 12, Align::right });

    // Header
    Row header = header_labels(columns);
    header[0] = emphasis(header[0]);
    header[1] = emphasis(header[1]);
    header[2] = muted(fmt::format("{}", "\xcf\x83"));
    header[3] = emphasis(header[3]);

    if (show_vi3) header[4] = emphasis(header[4]);

    if (show_vi1) header[show_vi3 ? 5 : 4] = emphasis(header[show_vi3 ? 5 : 4]);

    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    // Data rows
    for (int rank = 0; rank < rows; ++rank) {
      int j         = order[static_cast<std::size_t>(rank)];
      std::string v = fmt::format("x{}", j + 1);

      Row cells = {
        fmt::format("{}", rank + 1),
        v,
        fmt::format("{:.4f}", scale(j)),
        fmt::format("{:.6f}", vi2(j)),
      };

      if (show_vi3) cells.push_back(fmt::format("{:.6f}", vi3(j)));

      if (show_vi1) cells.push_back(fmt::format("{:.6f}", vi1(j)));

      out.println("{}", format_row(columns, cells));
    }

    if (rows < p) {
      out.println("{}", muted(fmt::format("... {} more variables not shown", p - rows)));
    }

    bool all_ones = (scale.array() - types::Feature(1)).abs().maxCoeff() < types::Feature(1e-6);

    if (!all_ones) {
      out.newline();
      out.println("{}", warning(
        "Note: VI was calculated using scaled coefficients (|a_j| * sigma_j).\n"
        "Variable contributions can only be theoretically interpreted as such\n"
        "if the model was trained on scaled data. Scaling also changes the\n"
        "projection-pursuit optimization, which may affect the resulting tree."));
    }

    out.newline();
  }

  void print_confusion_matrix(pptree::io::Output& out, const stats::ConfusionMatrix& cm) {
    using namespace pptree::io;

    auto class_err = cm.class_errors();

    out.println("{}", emphasis("Confusion Matrix:"));
    out.newline();

    // Header row: class labels
    std::string header = "    ";
    for (const auto& [label, idx] : cm.label_index) {
      header += fmt::format("{:>5}", label);
    }
    header += "  Error";
    out.println("{}", header);

    // Data rows
    for (const auto& [label, row_idx] : cm.label_index) {
      std::string row = fmt::format("{:>4}", label);

      for (const auto& [col_label, col_idx] : cm.label_index) {
        int val          = cm.values(row_idx, col_idx);
        std::string cell = fmt::format("{:>4}", val);

        if (row_idx == col_idx) {
          row += " " + success(cell);
        } else if (val > 0) {
          row += " " + error(cell);
        } else {
          row += " " + muted(cell);
        }
      }

      row += fmt::format("  {:.1f}%", class_err[row_idx] * 100);
      out.println("{}", row);
    }

    out.newline();
  }
}
