#pragma once

#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief Validate that a JSON object contains only expected keys.
   *
   * @param j        The JSON object to check.
   * @param context  Human-readable strategy description (e.g. "PDA").
   * @param allowed  Set of allowed key names.
   * @throws std::runtime_error if an unknown key is found.
   */
  inline void
  validate_json_keys(nlohmann::json const& j, std::string const& context, std::initializer_list<std::string> allowed) {
    std::set<std::string> allowed_set(allowed);

    for (auto it = j.begin(); it != j.end(); ++it) {
      if (allowed_set.find(it.key()) == allowed_set.end()) {
        throw std::runtime_error("Unknown " + context + " parameter: " + it.key());
      }
    }
  }
}
