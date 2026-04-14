/**
 * @file JsonApply.hpp
 * @brief Internal JSON-to-field helpers for CLI param deserialization.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace ppforest2::cli {
  /** @brief Set field from JSON if key exists. */
  template<typename T> void apply(nlohmann::json const& obj, std::string const& key, T& field) {
    if (obj.contains(key)) {
      field = obj[key].get<T>();
    }
  }

  /** @brief Set optional field from JSON if key exists. */
  template<typename T> void apply(nlohmann::json const& obj, std::string const& key, std::optional<T>& field) {
    if (obj.contains(key)) {
      field = obj[key].get<T>();
    }
  }
}
