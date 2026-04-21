#pragma once

#include <nlohmann/json.hpp>

namespace ppforest2::serialization {
  /**
   * @brief Validate the structural shape of a loaded model Export JSON.
   *
   * Runs before any `from_json` decoding so failures surface with
   * domain-specific, path-annotated messages instead of nlohmann's
   * cryptic type errors.
   *
   * Concentrates every cross-field and required-field check in one
   * place. Strategy internals are *not* validated here — each strategy's
   * own `from_json` performs that via `JsonReader`. This keeps the
   * validator responsibilities flat (skeleton + `config` + `meta`) and
   * avoids turning it into a schema library.
   *
   * Use `validate_tree_export` or `validate_forest_export` when the
   * caller already knows which variant it is loading; `validate_model_export`
   * accepts either.
   *
   * @throws std::runtime_error with a dotted path on validation failure.
   */
  void validate_model_export(nlohmann::json const& j);

  /** @brief Like `validate_model_export`, and assert `model_type == "tree"`. */
  void validate_tree_export(nlohmann::json const& j);

  /** @brief Like `validate_model_export`, and assert `model_type == "forest"`. */
  void validate_forest_export(nlohmann::json const& j);
}
