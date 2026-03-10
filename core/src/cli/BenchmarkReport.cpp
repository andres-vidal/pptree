/**
 * @file BenchmarkReport.cpp
 * @brief Benchmark result table formatting, comparison display, and export.
 */
#include "cli/BenchmarkReport.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
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

  std::string format_delta(double current, double baseline, int width = 8) {
    if (baseline <= 0) return fmt::format("{:>{}s}", "", width);

    double delta_pct = ((current - baseline) / baseline) * 100.0;
    std::string sign = delta_pct >= 0 ? "+" : "";
    std::string text = fmt::format("{:>{}s}", fmt::format("{}{:.1f}%", sign, delta_pct), width);

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

    // Header
    fmt::print("\n{}", emphasis(current.suite_name));

    if (!current.timestamp.empty()) {
      fmt::print(" {}", muted("(" + current.timestamp + ")"));
    }

    fmt::print("\n");

    if (has_baseline && !baseline->timestamp.empty()) {
      fmt::print("Baseline: {}\n", muted(baseline->timestamp));
    }

    fmt::print("\n");

    // Column headers — apply styling after padding to avoid ANSI codes
    // breaking fmt's width calculations.
    if (has_baseline) {
      fmt::print("  {} {:>6s} {:>4s} {:>3s} {:>6s} {:>5s} {:>5s} {:>18s} {:>8s} {:>12s} {:>8s} {:>8s}\n",
      emphasis(fmt::format("{:<20s}", "Scenario")), "n", "p", "g", "trees", "vars", "iters",
      "Time (ms)", muted(fmt::format("{:>8s}", "delta")), "Peak RSS", muted(fmt::format("{:>8s}", "delta")), "Test Err");
    } else {
      fmt::print("  {} {:>6s} {:>4s} {:>3s} {:>6s} {:>5s} {:>5s} {:>18s} {:>12s} {:>8s}\n",
      emphasis(fmt::format("{:<20s}", "Scenario")), "n", "p", "g", "trees", "vars", "iters",
      "Time (ms)", "Peak RSS", "Test Err");
    }

    // Separator
    int sep_width = has_baseline ? 120 : 100;
    std::string sep(sep_width, '-');
    fmt::print("  {}\n", muted(sep));

    // Rows — styled strings (vars_str, deltas) are pre-padded before
    // styling so ANSI codes don't break column alignment.
    for (const auto& r : current.results) {
      std::string time_str = fmt::format("{:>18s}", fmt::format("{:.1f} +/- {:.1f}", r.mean_time_ms, r.std_time_ms));
      std::string vars_str = r.trees > 0
        ? fmt::format("{:>5s}", fmt::format("{:.2f}", r.vars))
        : muted(fmt::format("{:>5s}", "--"));
      std::string rss_str = fmt::format("{:>12s}", format_rss(r.peak_rss_mb));
      std::string err_str = fmt::format("{:>8s}", fmt::format("{:.1f}%", r.mean_te_error * 100));

      if (has_baseline) {
        auto it                = baseline_index.find(r.name);
        std::string time_delta = fmt::format("{:>8s}", "");
        std::string rss_delta  = fmt::format("{:>8s}", "");

        if (it != baseline_index.end()) {
          time_delta = format_delta(r.mean_time_ms, it->second->mean_time_ms);

          if (r.peak_rss_mb >= 0 && it->second->peak_rss_mb >= 0) {
            rss_delta = format_delta(r.peak_rss_mb, it->second->peak_rss_mb);
          }
        }

        fmt::print("  {:<20s} {:>6d} {:>4d} {:>3d} {:>6d} {} {:>5d} {} {} {} {} {}\n",
        r.name, r.n, r.p, r.g, r.trees, vars_str, r.runs,
        time_str, time_delta, rss_str, rss_delta, err_str);
      } else {
        fmt::print("  {:<20s} {:>6d} {:>4d} {:>3d} {:>6d} {} {:>5d} {} {} {}\n",
        r.name, r.n, r.p, r.g, r.trees, vars_str, r.runs,
        time_str, rss_str, err_str);
      }
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

    // Table header
    if (has_baseline) {
      out << "| Scenario | n | p | g | trees | vars | iters | Time (ms) | \xCE\x94 Time | Peak RSS | \xCE\x94 RSS | Test Err |\n";
      out << "|----------|--:|--:|--:|------:|-----:|------:|----------:|------:|----------:|-----:|---------:|\n";
    } else {
      out << "| Scenario | n | p | g | trees | vars | iters | Time (ms) | Peak RSS | Test Err |\n";
      out << "|----------|--:|--:|--:|------:|-----:|------:|----------:|----------:|---------:|\n";
    }

    // Rows
    for (const auto& r : current.results) {
      std::string time_str = fmt::format("{:.1f} \xC2\xB1 {:.1f}", r.mean_time_ms, r.std_time_ms);
      std::string vars_str = r.trees > 0 ? fmt::format("{:.2f}", r.vars) : "--";
      std::string rss_str  = format_rss(r.peak_rss_mb);
      std::string err_str  = fmt::format("{:.1f}%", r.mean_te_error * 100);

      if (has_baseline) {
        auto it                = baseline_index.find(r.name);
        std::string time_delta = "";
        std::string rss_delta  = "";

        if (it != baseline_index.end()) {
          time_delta = format_delta_markdown(r.mean_time_ms, it->second->mean_time_ms);

          if (r.peak_rss_mb >= 0 && it->second->peak_rss_mb >= 0) {
            rss_delta = format_delta_markdown(r.peak_rss_mb, it->second->peak_rss_mb);
          }
        }

        out << fmt::format("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
        r.name, r.n, r.p, r.g, r.trees, vars_str, r.runs,
        time_str, time_delta, rss_str, rss_delta, err_str);
      } else {
        out << fmt::format("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
        r.name, r.n, r.p, r.g, r.trees, vars_str, r.runs,
        time_str, rss_str, err_str);
      }
    }

    out << fmt::format("\n{} scenarios completed in {:.1f}s\n",
    current.results.size(), current.total_time_ms / 1000.0);

    return out.str();
  }

  void write_results_csv(const SuiteResult& result, const std::string& path) {
    std::ofstream file(path);

    invariant(file.is_open(), fmt::format("Failed to open CSV file for writing: {}", path));

    // Header
    file << "scenario,n,p,g,trees,vars,runs,mean_time_ms,std_time_ms,"
         << "mean_train_error,mean_test_error,peak_rss_mb,scenario_time_ms\n";

    // Data rows
    for (const auto& r : result.results) {
      file << fmt::format("{},{},{},{},{},{:.2f},{},{:.2f},{:.2f},{:.4f},{:.4f},{:.1f},{:.0f}\n",
      r.name, r.n, r.p, r.g, r.trees, r.vars, r.runs,
      r.mean_time_ms, r.std_time_ms,
      r.mean_tr_error, r.mean_te_error,
      r.peak_rss_mb, r.scenario_time_ms);
    }
  }
}
