/**
 * @file IO.hpp
 * @brief File I/O utilities, CSV reading/writing, and peak RSS measurement.
 */
#pragma once

#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::io {
  // --- File output helpers ---

  /**
   * @brief Ensure a file path ends with the ".json" extension.
   * @param path The original file path.
   * @return The path with ".json" appended if it was missing.
   */
  std::string ensure_json_extension(const std::string& path);

  /**
   * @brief Exit with an error if a file already exists at the given path.
   * @param path The file path to check.
   */
  void check_file_not_exists(const std::string& path);

  /**
   * @brief Exit with an error if a directory already exists at the given path.
   * @param path The directory path to check.
   */
  void check_dir_not_exists(const std::string& path);

  /**
   * @brief Write a JSON object to a file (pretty-printed with indent 2).
   * @param data The JSON object to serialize.
   * @param path The output file path.
   */
  void write_json_file(const nlohmann::json& data, const std::string& path);

  // --- Peak RSS measurement ---

  /**
   * @brief Get the peak resident set size (RSS) of the current process in bytes.
   *
   * Uses platform-specific APIs: `GetProcessMemoryInfo` on Windows,
   * `getrusage` on POSIX (macOS reports bytes, Linux reports KB * 1024).
   *
   * @return Peak RSS in bytes, or -1 on failure.
   */
  long get_peak_rss_bytes();

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
  stats::DataPacket read_csv(const std::string& filename);

  /**
   * @brief Read a CSV file and sort rows so that response groups are contiguous.
   *
   * Calls read_csv() and then sorts the data if the response vector is not
   * already contiguous, as required by the training routines.
   *
   * @param filename Path to the CSV file.
   * @return A DataPacket with contiguous group ordering.
   */
  stats::DataPacket read_csv_sorted(const std::string& filename);

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  void write_csv(const stats::DataPacket& data, const std::string& filename);
}
