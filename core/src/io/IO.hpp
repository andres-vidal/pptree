/**
 * @file IO.hpp
 * @brief File I/O utilities, CSV reading/writing, and peak RSS measurement.
 */
#pragma once

#include "csv.hpp"
#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace pptree::io {
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
      fmt::print(stderr, "Error: File already exists: {}\n", path);
      std::exit(1);
    }
  }

  /**
   * @brief Exit with an error if a directory already exists at the given path.
   * @param path The directory path to check.
   */
  inline void check_dir_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "Error: Directory already exists: {}\n", path);
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
  inline stats::DataPacket read_csv(const std::string& filename) {
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

    return stats::DataPacket(x, y);
  }

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  inline void write_csv(const stats::DataPacket& data, const std::string& filename) {
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
}
