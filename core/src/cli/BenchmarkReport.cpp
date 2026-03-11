/**
 * @file BenchmarkReport.cpp
 * @brief Benchmark result table formatting, comparison display, and export.
 */
#include "cli/BenchmarkReport.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "io/Table.hpp"
#include "utils/Invariant.hpp"

#include <fmt/format.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>

namespace pptree::cli {
namespace {
  std::string format_rss(double mb) {
    if (mb < 0) return "N/A";

    return fmt::format("{:.1f} MB", mb);
  }

  std::string format_delta(double current, double baseline) {
    if (baseline <= 0) return "";

    double delta_pct = ((current - baseline) / baseline) * 100.0;
    std::string sign = delta_pct >= 0 ? "+" : "";
    std::string text = fmt::format("{}{:.1f}%", sign, delta_pct);

    // For time and RSS: negative = improvement (green), positive = regression (red)
    if (delta_pct < -1.0) {
      return pptree::io::success(text);
    } else if (delta_pct > 1.0) {
      return pptree::io::error(text);
    } else {
      return pptree::io::muted(text);
    }
  }

  std::string format_delta_markdown(double current, double baseline) {
    if (baseline <= 0) return "";

    double delta_pct = ((current - baseline) / baseline) * 100.0;
    std::string sign = delta_pct >= 0 ? "+" : "";
    std::string text = fmt::format("{}{:.1f}%", sign, delta_pct);

    if (delta_pct < -1.0) {
      return fmt::format("\xF0\x9F\x9F\xA2 {}", text);
    } else if (delta_pct > 1.0) {
      return fmt::format("\xF0\x9F\x94\xB4 {}", text);
    } else {
      return fmt::format("\xE2\x9A\xAA {}", text);
    }
  }

  std::unordered_map<std::string, const ScenarioResult *>
  build_baseline_index(const SuiteResult& baseline) {
    std::unordered_map<std::string, const ScenarioResult *> index;

    for (const auto& r : baseline.results) {
      index[r.name] = &r;
    }

    return index;
  }
}

  void print_benchmark_table(
    const SuiteResult&                current,
    const std::optional<SuiteResult>& baseline) {
    using namespace pptree::io;

    bool has_baseline   = baseline.has_value();
    auto baseline_index = has_baseline ? build_baseline_index(*baseline) : decltype(build_baseline_index(*baseline)){};

    // Title
    fmt::print("\n{}", emphasis(current.suite_name));

    if (!current.timestamp.empty()) {
      fmt::print(" {}", muted("(" + current.timestamp + ")"));
    }

    fmt::print("\n");

    if (has_baseline && !baseline->timestamp.empty()) {
      fmt::print("Baseline: {}\n", muted(baseline->timestamp));
    }

    fmt::print("\n");

    // Build columns conditionally
    std::vector<Column> columns = {
      { "Scenario",  20, Align::left },
      { "n",          6, Align::right },
      { "p",          4, Align::right },
      { "g",          3, Align::right },
      { "trees",      6, Align::right },
      { "vars",       5, Align::right },
      { "split",      5, Align::right },
      { "iters",      5, Align::right },
      { "Time (ms)", 18, Align::right },
    };

    if (has_baseline) columns.push_back({ "delta", 8, Align::right });

    columns.push_back({ "Peak RSS", 12, Align::right });

    if (has_baseline) columns.push_back({ "delta", 8, Align::right });

    columns.push_back({ "Train Err", 9, Align::right });

    if (has_baseline) columns.push_back({ "delta", 8, Align::right });

    columns.push_back({ "Test Err", 8, Align::right });

    if (has_baseline) columns.push_back({ "delta", 8, Align::right });

    // Header — style Scenario and delta labels
    Row header = header_labels(columns);
    header[0] = emphasis(header[0]);

    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == "delta") header[i] = muted("delta");
    }

    fmt::print("  {}\n", format_row(columns, header));
    fmt::print("  {}\n", muted(format_separator(columns)));

    // Data rows
    for (const auto& r : current.results) {
      std::string time_str = fmt::format("{:.1f} +/- {:.1f}", r.mean_time_ms, r.std_time_ms);
      std::string vars_str = r.trees > 0
        ? fmt::format("{:.2f}", r.vars)
        : muted("--");
      std::string split_str  = fmt::format("{}%", static_cast<int>(r.train_ratio * 100));
      std::string rss_str    = format_rss(r.peak_rss_mb);
      std::string tr_err_str = fmt::format("{:.1f}%", r.mean_tr_error * 100);
      std::string te_err_str = fmt::format("{:.1f}%", r.mean_te_error * 100);

      Row cells = {
        r.name,
        fmt::format("{}", r.n),
        fmt::format("{}", r.p),
        fmt::format("{}", r.g),
        fmt::format("{}", r.trees),
        vars_str,
        split_str,
        fmt::format("{}", r.runs),
        time_str,
      };

      std::string time_delta, rss_delta, tr_err_delta, te_err_delta;

      if (has_baseline) {
        auto it = baseline_index.find(r.name);

        if (it != baseline_index.end()) {
          time_delta = format_delta(r.mean_time_ms, it->second->mean_time_ms);

          if (r.peak_rss_mb >= 0 && it->second->peak_rss_mb >= 0) {
            rss_delta = format_delta(r.peak_rss_mb, it->second->peak_rss_mb);
          }

          if (it->second->mean_tr_error > 0) {
            tr_err_delta = format_delta(r.mean_tr_error, it->second->mean_tr_error);
          }

          if (it->second->mean_te_error > 0) {
            te_err_delta = format_delta(r.mean_te_error, it->second->mean_te_error);
          }
        }

        cells.push_back(time_delta);
      }

      cells.push_back(rss_str);

      if (has_baseline) {
        cells.push_back(rss_delta);
      }

      cells.push_back(tr_err_str);

      if (has_baseline) {
        cells.push_back(tr_err_delta);
      }

      cells.push_back(te_err_str);

      if (has_baseline) {
        cells.push_back(te_err_delta);
      }

      fmt::print("  {}\n", format_row(columns, cells));
    }

    // Footer
    fmt::print("\n  {} scenarios completed in {:.1f}s\n\n",
    emphasis(std::to_string(current.results.size())),
    current.total_time_ms / 1000.0);
  }

  void write_results_json(const SuiteResult& result, const std::string& path) {
    pptree::io::write_json_file(result.to_json(), path);
  }

  std::string format_benchmark_markdown(
    const SuiteResult&                current,
    const std::optional<SuiteResult>& baseline) {
    using namespace pptree::io;

    bool has_baseline   = baseline.has_value();
    auto baseline_index = has_baseline ? build_baseline_index(*baseline) : decltype(build_baseline_index(*baseline)){};

    std::ostringstream out;

    out << "## " << current.suite_name << "\n\n";

    if (!current.timestamp.empty()) {
      out << "Current: " << current.timestamp << "\n";
    }

    if (has_baseline && !baseline->timestamp.empty()) {
      out << "Baseline: " << baseline->timestamp << "\n";
    }

    out << "\n";

    // Build columns conditionally
    std::vector<Column> columns = {
      { "Scenario",  0, Align::left },
      { "n",         0, Align::right },
      { "p",         0, Align::right },
      { "g",         0, Align::right },
      { "trees",     0, Align::right },
      { "vars",      0, Align::right },
      { "split",     0, Align::right },
      { "iters",     0, Align::right },
      { "Time (ms)", 0, Align::right },
    };

    if (has_baseline) columns.push_back({ "\xCE\x94 Time", 0, Align::right });

    columns.push_back({ "Peak RSS", 0, Align::right });

    if (has_baseline) columns.push_back({ "\xCE\x94 RSS", 0, Align::right });

    columns.push_back({ "Train Err", 0, Align::right });

    if (has_baseline) columns.push_back({ "\xCE\x94 Train", 0, Align::right });

    columns.push_back({ "Test Err", 0, Align::right });

    if (has_baseline) columns.push_back({ "\xCE\x94 Test", 0, Align::right });

    out << format_md_row(header_labels(columns)) << "\n";
    out << format_md_separator(columns) << "\n";

    // Data rows
    for (const auto& r : current.results) {
      std::string time_str   = fmt::format("{:.1f} \xC2\xB1 {:.1f}", r.mean_time_ms, r.std_time_ms);
      std::string vars_str   = r.trees > 0 ? fmt::format("{:.2f}", r.vars) : "--";
      std::string split_str  = fmt::format("{}%", static_cast<int>(r.train_ratio * 100));
      std::string rss_str    = format_rss(r.peak_rss_mb);
      std::string tr_err_str = fmt::format("{:.1f}%", r.mean_tr_error * 100);
      std::string te_err_str = fmt::format("{:.1f}%", r.mean_te_error * 100);

      Row cells = {
        r.name,
        fmt::format("{}", r.n),
        fmt::format("{}", r.p),
        fmt::format("{}", r.g),
        fmt::format("{}", r.trees),
        vars_str,
        split_str,
        fmt::format("{}", r.runs),
        time_str,
      };

      std::string time_delta, rss_delta, tr_err_delta, te_err_delta;

      if (has_baseline) {
        auto it = baseline_index.find(r.name);

        if (it != baseline_index.end()) {
          time_delta = format_delta_markdown(r.mean_time_ms, it->second->mean_time_ms);

          if (r.peak_rss_mb >= 0 && it->second->peak_rss_mb >= 0) {
            rss_delta = format_delta_markdown(r.peak_rss_mb, it->second->peak_rss_mb);
          }

          if (it->second->mean_tr_error > 0) {
            tr_err_delta = format_delta_markdown(r.mean_tr_error, it->second->mean_tr_error);
          }

          if (it->second->mean_te_error > 0) {
            te_err_delta = format_delta_markdown(r.mean_te_error, it->second->mean_te_error);
          }
        }

        cells.push_back(time_delta);
      }

      cells.push_back(rss_str);

      if (has_baseline) {
        cells.push_back(rss_delta);
      }

      cells.push_back(tr_err_str);

      if (has_baseline) {
        cells.push_back(tr_err_delta);
      }

      cells.push_back(te_err_str);

      if (has_baseline) {
        cells.push_back(te_err_delta);
      }

      out << format_md_row(cells) << "\n";
    }

    out << fmt::format("\n{} scenarios completed in {:.1f}s\n",
    current.results.size(), current.total_time_ms / 1000.0);

    return out.str();
  }

  void write_results_csv(const SuiteResult& result, const std::string& path) {
    std::ofstream file(path);

    invariant(file.is_open(), fmt::format("Failed to open CSV file for writing: {}", path));

    // Header
    file << "scenario,n,p,g,trees,vars,train_ratio,runs,mean_time_ms,std_time_ms,"
         << "mean_train_error,mean_test_error,peak_rss_mb,scenario_time_ms\n";

    // Data rows
    for (const auto& r : result.results) {
      file << fmt::format("{},{},{},{},{},{:.2f},{:.2f},{},{:.2f},{:.2f},{:.4f},{:.4f},{:.1f},{:.0f}\n",
      r.name, r.n, r.p, r.g, r.trees, r.vars, r.train_ratio, r.runs,
      r.mean_time_ms, r.std_time_ms,
      r.mean_tr_error, r.mean_te_error,
      r.peak_rss_mb, r.scenario_time_ms);
    }
  }
}
