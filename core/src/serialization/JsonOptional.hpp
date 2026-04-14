/**
 * @file JsonOptional.hpp
 * @brief nlohmann::json support for std::optional<T>.
 *
 * Include this header alongside <nlohmann/json.hpp> to enable
 * transparent JSON conversion for std::optional fields:
 *
 * @code
 *   j["key"] = std::optional<int>(42);   // writes 42
 *   j["key"] = std::optional<int>{};     // writes null
 *   auto opt = j["key"].get<std::optional<int>>();  // reads back
 * @endcode
 */
#pragma once

#include <nlohmann/json.hpp>
#include <optional>

namespace nlohmann {
  template<typename T> struct adl_serializer<std::optional<T>> {
    static void to_json(json& j, std::optional<T> const& opt) {
      if (opt) {
        j = *opt;
      } else {
        j = nullptr;
      }
    }

    static void from_json(json const& j, std::optional<T>& opt) {
      if (j.is_null()) {
        opt = std::nullopt;
      } else {
        opt = j.get<T>();
      }
    }
  };
}
