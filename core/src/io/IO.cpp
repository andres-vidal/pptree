/**
 * @file IO.cpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#include "io/IO.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/UserError.hpp"

#include "csv.hpp" // IWYU pragma: keep

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace ppforest2::io {
  void check_file_exists(std::string const& path) {
    user_error(std::filesystem::exists(path), fmt::format("File not found: {}", path));
  }

  void check_file_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("File already exists: {}", path));
  }

  void check_dir_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("Directory already exists: {}", path));
  }
}

namespace ppforest2::io::json {
  std::string ensure_extension(std::string const& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  nlohmann::json read_file(std::string const& path, ErrorHandler on_error) {
    std::ifstream in(path);

    on_error(in.is_open(), fmt::format("Could not open file: {}", path));

    try {
      return nlohmann::json::parse(in);
    } catch (nlohmann::json::parse_error const& e) {
      on_error(false, fmt::format("Invalid JSON in file: {}", e.what()));
      throw; // unreachable — on_error always throws
    }
  }

  void write_file(nlohmann::json const& data, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << data.dump(2);
    out.close();
  }
}

namespace ppforest2::io::text {
  void write_file(std::string const& content, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << content;
    out.close();
  }
}

namespace ppforest2::io::csv {
  namespace {
    bool is_numeric(std::string const& s) {
      if (s.empty()) {
        return false;
      }

      char* end = nullptr;
      std::strtod(s.c_str(), &end);
      return end != s.c_str() && *end == '\0';
    }

    struct CategoricalEncoder {
      std::unordered_map<std::string, int> mapping;

      types::Feature encode(std::string const& value) {
        auto it = mapping.find(value);

        if (it == mapping.end()) {
          int code       = static_cast<int>(mapping.size());
          mapping[value] = code;
          return static_cast<types::Feature>(code);
        }

        return static_cast<types::Feature>(it->second);
      }
    };
  }

  stats::DataPacket read(std::string const& filename) {
    ::csv::CSVReader reader(filename);

    // Extract feature column names from header (all columns except the last).
    auto col_names = reader.get_col_names();
    std::vector<std::string> feature_names;

    if (!col_names.empty()) {
      feature_names.assign(col_names.begin(), col_names.end() - 1);
    }

    // First pass: read all raw string values to detect categorical columns.
    std::vector<std::vector<std::string>> raw_rows;
    int n_cols = 0;

    for (::csv::CSVRow& row : reader) {
      int row_num = static_cast<int>(raw_rows.size()) + 1;
      ppforest2::user_error(
          row.size() >= 2,
          fmt::format("Row {} has only {} column(s) — expected at least 2 (features + label)", row_num, row.size())
      );

      std::vector<std::string> values;

      for (int j = 0; j < row.size(); ++j) {
        values.push_back(row[j].get<std::string>());
      }

      if (n_cols == 0) {
        n_cols = static_cast<int>(values.size());
      }

      raw_rows.push_back(std::move(values));
    }

    ppforest2::user_error(!raw_rows.empty(), "CSV file is empty or has no data rows");

    int const n_features = n_cols - 1;

    // Detect which feature columns are categorical (non-numeric).
    std::vector<bool> is_categorical(static_cast<std::size_t>(n_features), false);

    for (int j = 0; j < n_features; ++j) {
      for (auto const& row : raw_rows) {
        if (!is_numeric(row[static_cast<std::size_t>(j)])) {
          is_categorical[static_cast<std::size_t>(j)] = true;
          break;
        }
      }
    }

    // Encode features (numeric or categorical) and response labels.
    std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));

    int const n = static_cast<int>(raw_rows.size());

    types::FeatureMatrix x(n, n_features);
    types::GroupIdVector y_int(n);

    std::unordered_map<std::string, int> label_mapping;
    std::vector<std::string> label_names;

    for (int i = 0; i < n; ++i) {
      auto const& row = raw_rows[static_cast<std::size_t>(i)];
      ppforest2::user_error(
          static_cast<int>(row.size()) == n_cols,
          fmt::format("Row {} has {} column(s), expected {} (same as row 1)", i + 1, row.size(), n_cols)
      );

      for (int j = 0; j < n_features; ++j) {
        auto const& val = row[static_cast<std::size_t>(j)];

        if (is_categorical[static_cast<std::size_t>(j)]) {
          x(i, j) = encoders[static_cast<std::size_t>(j)].encode(val);
        } else {
          x(i, j) = std::stof(val);
        }
      }

      // Outcome label (last column).
      std::string const& label_str = row[static_cast<std::size_t>(n_cols - 1)];

      auto [it, inserted] = label_mapping.try_emplace(label_str, static_cast<int>(label_names.size()));

      if (inserted) {
        label_names.push_back(label_str);
      }

      y_int[i] = label_mapping[label_str];
    }

    types::OutcomeVector y = y_int.cast<types::Outcome>();
    return stats::DataPacket(x, y, label_names, feature_names);
  }

  stats::DataPacket read_sorted(std::string const& filename) {
    stats::DataPacket data = read(filename);

    // Sort in place on `data.x` instead of copying it. `stats::sort` wants a
    // `GroupIdVector`, but we can't sort the `OutcomeVector` directly without
    // adding yet another overload; cast once up front, sort in place, and
    // write the permuted integer labels back through a single cast. If the
    // data is already group-contiguous, skip all of that and return as-is.
    types::GroupIdVector y_int = data.y.cast<types::GroupId>();

    if (!stats::GroupPartition::is_contiguous(y_int)) {
      stats::sort(data.x, y_int);
      data.y = y_int.cast<types::Outcome>();
    }

    return data;
  }

  stats::DataPacket read_regression_sorted(std::string const& filename) {
    ::csv::CSVReader reader(filename);

    auto col_names = reader.get_col_names();
    std::vector<std::string> feature_names;

    if (!col_names.empty()) {
      feature_names.assign(col_names.begin(), col_names.end() - 1);
    }

    std::vector<std::vector<std::string>> raw_rows;
    int n_cols = 0;

    for (::csv::CSVRow& row : reader) {
      int row_num = static_cast<int>(raw_rows.size()) + 1;
      ppforest2::user_error(
          row.size() >= 2,
          fmt::format("Row {} has only {} column(s) — expected at least 2 (features + response)", row_num, row.size())
      );

      std::vector<std::string> values;

      for (int j = 0; j < row.size(); ++j) {
        values.push_back(row[j].get<std::string>());
      }

      if (n_cols == 0) {
        n_cols = static_cast<int>(values.size());
      }

      raw_rows.push_back(std::move(values));
    }

    ppforest2::user_error(!raw_rows.empty(), "CSV file is empty or has no data rows");

    int const n_features = n_cols - 1;
    int const n          = static_cast<int>(raw_rows.size());

    // Detect categorical feature columns (same as classification reader).
    std::vector<bool> is_categorical(static_cast<std::size_t>(n_features), false);

    for (int j = 0; j < n_features; ++j) {
      for (auto const& row : raw_rows) {
        if (!is_numeric(row[static_cast<std::size_t>(j)])) {
          is_categorical[static_cast<std::size_t>(j)] = true;
          break;
        }
      }
    }

    std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));

    types::FeatureMatrix x(n, n_features);
    types::OutcomeVector y_cont(n);

    for (int i = 0; i < n; ++i) {
      auto const& row = raw_rows[static_cast<std::size_t>(i)];

      ppforest2::user_error(
          static_cast<int>(row.size()) == n_cols,
          fmt::format("Row {} has {} column(s), expected {} (same as row 1)", i + 1, row.size(), n_cols)
      );

      for (int j = 0; j < n_features; ++j) {
        auto const& val = row[static_cast<std::size_t>(j)];

        if (is_categorical[static_cast<std::size_t>(j)]) {
          x(i, j) = encoders[static_cast<std::size_t>(j)].encode(val);
        } else {
          x(i, j) = std::stof(val);
        }
      }

      std::string const& y_str = row[static_cast<std::size_t>(n_cols - 1)];

      ppforest2::user_error(
          is_numeric(y_str), fmt::format("Row {} response value '{}' is not numeric (regression mode)", i + 1, y_str)
      );

      y_cont(i) = std::stof(y_str);
    }

    // Sort by continuous y.
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&y_cont](int a, int b) { return y_cont(a) < y_cont(b); });

    types::FeatureMatrix sorted_x(n, n_features);
    types::OutcomeVector sorted_y(n);

    for (int i = 0; i < n; ++i) {
      sorted_x.row(i) = x.row(order[static_cast<std::size_t>(i)]);
      sorted_y(i)     = y_cont(order[static_cast<std::size_t>(i)]);
    }

    return stats::DataPacket(sorted_x, sorted_y, stats::DataPacket::NoGroups{}, feature_names);
  }

  void write(stats::DataPacket const& data, std::string const& filename) {
    std::ofstream out(filename);

    user_error(out.is_open(), fmt::format("Could not open file for writing: {}", filename));

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
