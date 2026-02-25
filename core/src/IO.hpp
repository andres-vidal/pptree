/**
 * @file IO.hpp
 * @brief File I/O utilities, CSV reading/writing, peak RSS measurement,
 *        and terminal announcement helpers for the pptree CLI.
 */
#pragma once

#include "csv.hpp"
#include <vector>
#include <random>
#include <unordered_map>
#include <fstream>
#include <numeric>
#include <cmath>
#include <filesystem>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "Stats.hpp"
#include "Color.hpp"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace pptree {
  // --- File output helpers ---

  /**
   * @brief Ensure a file path ends with the ".json" extension.
   * @param path The original file path.
   * @return The path with ".json" appended if it was missing.
   */
  inline std::string ensure_json_extension(const std::string& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  /**
   * @brief Exit with an error if a file already exists at the given path.
   * @param path The file path to check.
   */
  inline void check_file_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "{} File already exists: {}\n", error("Error:"), path);
      std::exit(1);
    }
  }

  /**
   * @brief Exit with an error if a directory already exists at the given path.
   * @param path The directory path to check.
   */
  inline void check_dir_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "{} Directory already exists: {}\n", error("Error:"), path);
      std::exit(1);
    }
  }

  /**
   * @brief Write a JSON object to a file (pretty-printed with indent 2).
   * @param data The JSON object to serialize.
   * @param path The output file path.
   */
  inline void write_json_file(const nlohmann::json& data, const std::string& path) {
    std::ofstream out(path);

    if (!out.is_open()) {
      fmt::print(stderr, "Error: Could not open file for writing: {}\n", path);
      std::exit(1);
    }

    out << data.dump(2);
    out.close();
  }

  // --- Peak RSS measurement ---

  /**
   * @brief Get the peak resident set size (RSS) of the current process in bytes.
   *
   * Uses platform-specific APIs: `GetProcessMemoryInfo` on Windows,
   * `getrusage` on POSIX (macOS reports bytes, Linux reports KB * 1024).
   *
   * @return Peak RSS in bytes, or -1 on failure.
   */
  inline long get_peak_rss_bytes() {
    #ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
      return static_cast<long>(pmc.PeakWorkingSetSize);
    }

    return -1;

    #else
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      #ifdef __APPLE__
      return usage.ru_maxrss;          // macOS: already in bytes

      #else
      return usage.ru_maxrss * 1024L;  // Linux: reported in KB

      #endif
    }

    return -1;

    #endif // ifdef _WIN32
  }

  // --- CSV I/O ---

  /**
   * @brief Read a CSV file into a DataPacket.
   *
   * Assumes the last column is the response variable (class label as string)
   * and all preceding columns are numeric features. String labels are mapped
   * to contiguous integer codes starting at 0.
   *
   * @param filename Path to the CSV file.
   * @return A DataPacket containing the feature matrix and response vector.
   * @throws std::runtime_error If the file is empty or has inconsistent columns.
   */
  DataPacket read_csv(const std::string& filename) {
    csv::CSVReader reader(filename);
    std::vector<std::vector<types::Feature> > featureData;
    std::vector<std::string> rawLabels;

    for (csv::CSVRow& row : reader) {
      if (row.size() < 1) {
        throw std::runtime_error("CSV row has no columns.");
      }

      std::vector<types::Feature> currentFeatures;
      for (int j = 0; j < row.size() - 1; ++j) {
        currentFeatures.push_back(row[j].get<types::Feature>());
      }

      featureData.push_back(std::move(currentFeatures));

      // Read the last column as a string.
      std::string labelStr = row[row.size() - 1].get<std::string>();
      rawLabels.push_back(labelStr);
    }

    if (featureData.empty()) {
      throw std::runtime_error("CSV file is empty.");
    }

    // Map string labels to integer codes.
    std::unordered_map<std::string, int> labelMapping;
    std::vector<int> labels;
    int labelIndex = 0;
    for (const auto &labelStr : rawLabels) {
      if (labelMapping.find(labelStr) == labelMapping.end()) {
        labelMapping[labelStr] = labelIndex++;
      }

      labels.push_back(labelMapping[labelStr]);
    }

    // Determine dimensions for feature matrix.
    const int n = featureData.size();
    const int p = featureData[0].size();

    types::FeatureMatrix x(n, p);
    for (int i = 0; i < n; ++i) {
      if (featureData[i].size() != p) {
        throw std::runtime_error("Inconsistent number of feature columns in CSV file.");
      }

      for (int j = 0; j < p; ++j) {
        x(i, j) = featureData[i][j];
      }
    }

    types::ResponseVector y(n);
    for (int i = 0; i < n; ++i) {
      y[i] = labels[i];
    }

    return DataPacket(x, y);
  }

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  void write_csv(const DataPacket& data, const std::string& filename) {
    std::ofstream out(filename);

    if (!out.is_open()) {
      fmt::print(stderr, "Error: Could not open file for writing: {}\n", filename);
      std::exit(1);
    }

    for (int i = 0; i < data.x.rows(); ++i) {
      for (int j = 0; j < data.x.cols(); ++j) {
        out << data.x(i, j);

        if (j < data.x.cols() - 1) {
          out << ",";
        }
      }

      out << "," << data.y[i] << "\n";
    }

    out.close();
  }

  // --- Announcement helpers ---

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
  void announce_configuration(
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
      return sd(tr_times);
    }

    double std_tr_error() const {
      return sd(tr_error);
    }

    double std_te_error() const {
      return sd(te_error);
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
  void announce_results(const ModelStats& stats) {
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
}
