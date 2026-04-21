#include "serialization/ExportValidation.hpp"

#include "utils/JsonReader.hpp"

#include <stdexcept>
#include <string>

namespace ppforest2::serialization {
  namespace {
    // Top-level is deliberately an open set. The writer emits the core
    // `model`/`config`/`meta` trio plus optional metrics (confusion
    // matrices, regression metrics, OOB error, VI) and CLI annotations
    // (`training_duration_ms`, `save_path`). Downstream tools (e.g. the
    // golden generator) add more annotations on top. Enforcing a closed
    // set would reject any such extension. Required-key presence is
    // still enforced below.

    constexpr auto MODES = {"classification", "regression"};

    constexpr auto CONFIG_REQUIRED_BASE = {
        "mode", "size", "seed", "threads", "max_retries",
        "pp", "vars", "cutpoint", "stop", "binarize", "grouping", "leaf"
    };

    // Validate the `config` block, returning the mode string.
    std::string validate_config(JsonReader const& config) {
      for (char const* key : CONFIG_REQUIRED_BASE) {
        if (!config.contains(key)) {
          throw std::runtime_error(
              config.path() + "." + key + ": missing required key"
          );
        }
      }
      std::string const mode = config.require_enum("mode", MODES);

      (void)config.require_int("size", 0);
      (void)config.require_int("seed");
      (void)config.require_int("threads", 0);
      (void)config.require_int("max_retries", 0);

      // `binarize` is required for every mode — regression specs carry
      // `binarize::Disabled` (a mode-agnostic placeholder) rather than
      // omitting the key. Enforced above via `CONFIG_REQUIRED_BASE`.

      return mode;
    }

    // Validate the `meta` block, cross-checking against the config mode.
    void validate_meta(JsonReader const& meta, std::string const& config_mode) {
      (void)meta.require_int("observations", 0);
      (void)meta.require_int("features", 0);

      if (meta.contains("mode")) {
        std::string const meta_mode = meta.require_enum("mode", MODES);
        if (meta_mode != config_mode) {
          throw std::runtime_error(
              meta.path() + ".mode: disagrees with config.mode (meta='" +
              meta_mode + "', config='" + config_mode + "')"
          );
        }
      }

      // `feature_names`, if present, must be an array of strings. Empty is
      // allowed — not every training path carries column names.
      if (meta.contains("feature_names")) {
        auto const& names = meta.require_array("feature_names");
        for (std::size_t i = 0; i < names.size(); ++i) {
          if (!names[i].is_string()) {
            throw std::runtime_error(
                meta.path() + ".feature_names[" + std::to_string(i) +
                "]: expected string, got " + names[i].type_name()
            );
          }
        }
      }

      // Mode/groups pairing: regression omits `meta.groups` entirely;
      // classification requires a non-empty array. See discussion in
      // `Export<Model::Ptr>::to_json` for the writer side of this contract.
      bool const is_regression = config_mode == "regression";
      bool const has_groups    = meta.contains("groups");

      if (is_regression && has_groups) {
        throw std::runtime_error(
            meta.path() + ".groups: must be absent for regression models (mode='regression')"
        );
      }
      if (!is_regression) {
        if (!has_groups) {
          throw std::runtime_error(
              meta.path() + ".groups: missing required key (classification models must provide it)"
          );
        }
        auto const& groups = meta.require_array("groups");
        if (groups.empty()) {
          throw std::runtime_error(
              meta.path() + ".groups: must be non-empty for classification models"
          );
        }
        for (std::size_t i = 0; i < groups.size(); ++i) {
          if (!groups[i].is_string()) {
            throw std::runtime_error(
                meta.path() + ".groups[" + std::to_string(i) +
                "]: expected string, got " + groups[i].type_name()
            );
          }
        }
      }
    }

    // Shared skeleton check. Returns the declared mode.
    std::string validate_skeleton(nlohmann::json const& j) {
      JsonReader const root(j, "Export");
      root.require_object();

      auto const config   = root.at("config");
      std::string const m = validate_config(config);
      auto const meta     = root.at("meta");
      validate_meta(meta, m);

      // `model_type`, if present, must be one of the known variants.
      if (root.contains("model_type")) {
        (void)root.require_enum("model_type", {"tree", "forest"});
      }
      return m;
    }
  }

  void validate_model_export(nlohmann::json const& j) {
    (void)validate_skeleton(j);

    JsonReader const root(j, "Export");
    auto const model = root.at("model");
    // The model block has one of two shapes — let the variant-specific
    // validators check further. Here we only assert it's an object.
    model.require_object();
  }

  void validate_tree_export(nlohmann::json const& j) {
    (void)validate_skeleton(j);

    JsonReader const root(j, "Export");
    if (root.contains("model_type")) {
      std::string const t = root.require_enum("model_type", {"tree", "forest"});
      if (t != "tree") {
        throw std::runtime_error(
            "Export.model_type: expected 'tree' (got '" + t + "')"
        );
      }
    }

    auto const model = root.at("model");
    if (!model.contains("root")) {
      throw std::runtime_error("Export.model.root: missing required key");
    }
  }

  void validate_forest_export(nlohmann::json const& j) {
    (void)validate_skeleton(j);

    JsonReader const root(j, "Export");
    if (root.contains("model_type")) {
      std::string const t = root.require_enum("model_type", {"tree", "forest"});
      if (t != "forest") {
        throw std::runtime_error(
            "Export.model_type: expected 'forest' (got '" + t + "')"
        );
      }
    }

    auto const model = root.at("model");
    (void)model.require_array("trees");
  }
}
