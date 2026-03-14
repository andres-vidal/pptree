/**
 * @file IO.cpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#include "io/IO.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include "csv.hpp"

#include <fmt/format.h>

#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>

namespace ppforest2::io {
  void check_file_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "Error: File already exists: {}\n", path);
      std::exit(1);
    }
  }

  void check_dir_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "Error: Directory already exists: {}\n", path);
      std::exit(1);
    }
  }
}

namespace ppforest2::io::json {
  std::string ensure_extension(const std::string& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  nlohmann::json read_file(const std::string& path) {
    std::ifstream in(path);

    if (!in.is_open()) {
      fmt::print(stderr, "Error: Could not open file: {}\n", path);
      std::exit(1);
    }

    try {
      return nlohmann::json::parse(in);
    } catch (const nlohmann::json::parse_error& e) {
      fmt::print(stderr, "Error: Invalid JSON in file: {}\n", e.what());
      std::exit(1);
    }
  }

  void write_file(const nlohmann::json& data, const std::string& path) {
    std::ofstream out(path);

    if (!out.is_open()) {
      fmt::print(stderr, "Error: Could not open file for writing: {}\n", path);
      std::exit(1);
    }

    out << data.dump(2);
    out.close();
  }
}

namespace ppforest2::io::csv {
  stats::DataPacket read(const std::string& filename) {
    ::csv::CSVReader reader(filename);
    std::vector<std::vector<types::Feature>> featureData;
    std::vector<std::string> rawLabels;

    int row_num = 0;

    for (::csv::CSVRow& row : reader) {
      ++row_num;
      ppforest2::user_error(row.size() >= 2, fmt::format("Row {} has only {} column(s) — expected at least 2 (features + label)", row_num, row.size()));

      std::vector<types::Feature> currentFeatures;
      for (int j = 0; j < row.size() - 1; ++j) {
        currentFeatures.push_back(row[j].get<types::Feature>());
      }

      featureData.push_back(std::move(currentFeatures));

      // Read the last column as a string.
      std::string labelStr = row[row.size() - 1].get<std::string>();
      rawLabels.push_back(labelStr);
    }

    ppforest2::user_error(!featureData.empty(), "CSV file is empty or has no data rows");

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
      ppforest2::user_error(featureData[i].size() == p, fmt::format("Row {} has {} feature column(s), expected {} (same as row 1)", i + 1, featureData[i].size(), p));

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

  std::vector<std::string> read_labels(const std::string& filename) {
    ::csv::CSVReader reader(filename);
    std::unordered_map<std::string, int> seen;
    std::vector<std::string> labels;

    for (::csv::CSVRow& row : reader) {
      std::string label = row[row.size() - 1].get<std::string>();

      if (seen.find(label) == seen.end()) {
        seen[label] = static_cast<int>(labels.size());
        labels.push_back(label);
      }
    }

    return labels;
  }

  stats::DataPacket read_sorted(const std::string& filename) {
    stats::DataPacket data = read(filename);

    types::FeatureMatrix x  = data.x;
    types::ResponseVector y = data.y;

    if (!stats::GroupPartition::is_contiguous(y)) {
      stats::sort(x, y);
    }

    return stats::DataPacket(x, y);
  }

  void write(const stats::DataPacket& data, const std::string& filename) {
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
