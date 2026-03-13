/**
 * @file IO.hpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#pragma once

#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::io {
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
}

namespace ppforest2::io::json {
  /**
   * @brief Ensure a file path ends with the ".json" extension.
   * @param path The original file path.
   * @return The path with ".json" appended if it was missing.
   */
  std::string ensure_extension(const std::string& path);

  /**
   * @brief Read a JSON file and parse its contents.
   * @param path The input file path.
   * @return The parsed JSON object.
   */
  nlohmann::json read_file(const std::string& path);

  /**
   * @brief Write a JSON object to a file (pretty-printed with indent 2).
   * @param data The JSON object to serialize.
   * @param path The output file path.
   */
  void write_file(const nlohmann::json& data, const std::string& path);
}

namespace ppforest2::io::csv {
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
  stats::DataPacket read(const std::string& filename);

  /**
   * @brief Read a CSV file and sort rows so that response groups are contiguous.
   *
   * Calls read() and then sorts the data if the response vector is not
   * already contiguous, as required by the training routines.
   *
   * @param filename Path to the CSV file.
   * @return A DataPacket with contiguous group ordering.
   */
  stats::DataPacket read_sorted(const std::string& filename);

  /**
   * @brief Read class label names from a CSV file in the order they first appear.
   *
   * Returns a vector where index i is the string label that maps to integer code i,
   * matching the encoding produced by read().
   *
   * @param filename Path to the CSV file.
   * @return A vector of class label strings, ordered by their integer code.
   */
  std::vector<std::string> read_labels(const std::string& filename);

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  void write(const stats::DataPacket& data, const std::string& filename);
}
