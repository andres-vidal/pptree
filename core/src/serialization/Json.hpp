#pragma once

#include "models/TreeNode.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/Tree.hpp"
#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>

/**
 * @brief JSON serialization and deserialization for ppforest2 models.
 *
 * Uses nlohmann/json.  Provides to_json() for serializing trees,
 * forests, confusion matrices, and variable importance to JSON.
 * Deserialization uses `j.get<T>()` via nlohmann ADL:
 *
 * @code
 *   // Serialize and write to file:
 *   auto j = serialization::to_json(forest);
 *   io::json::write_file(j, "model.json");
 *
 *   // Read a full export (model + config + meta):
 *   auto j2 = io::json::read_file("model.json");
 *   auto e  = j2.get<Export<Forest>>();  // e.model, e.groups, e.spec
 *
 *   // Read bare model block (integer labels):
 *   Tree tree = model_json.get<Tree>();
 * @endcode
 */
namespace ppforest2::serialization {
  using json = nlohmann::json;

  /** @brief Group name vector for labeled serialization. */
  using GroupNames = std::vector<std::string>;

  /**
   * @brief A model bundled with its export metadata and optional metrics.
   *
   * Represents the full JSON export format:
   * `{ model_type, model, config, meta, [metrics] }`.
   *
   * Use `j.get<Export<Tree>>()` or `j.get<Export<Model::Ptr>>()` to
   * deserialize a full export, and `model_export.to_json()` to serialize.
   *
   * For `Export<Model::Ptr>`, construct with training data to compute
   * metrics automatically, or use `compute_metrics()` after construction.
   */
  template<typename T>
  struct Export {
    T model;
    GroupNames groups;
    TrainingSpec::Ptr spec;
    int n_observations = 0;
    int n_features     = 0;
    std::vector<std::string> feature_names;

    /** @name Optional metrics — serialized by to_json() when present. */
    ///@{
    std::optional<VariableImportance>       variable_importance;
    std::optional<stats::ConfusionMatrix>   training_confusion_matrix;
    std::optional<stats::ConfusionMatrix>   oob_confusion_matrix;
    std::optional<double>                   oob_error;
    ///@}

    /** @brief Serialize to JSON. Only defined for Export<Model::Ptr>. */
    json to_json() const;

    /**
     * @brief Compute and store metrics on this export.
     *
     * Only defined for Export<Model::Ptr>.
     */
    void compute_metrics(
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y);
  };

  // Explicit specialization declarations for Export<Model::Ptr>.
  template<> json Export<Model::Ptr>::to_json() const;
  template<> void Export<Model::Ptr>::compute_metrics(
    const types::FeatureMatrix&, const types::ResponseVector&);

  /** @brief Visitor that serializes a tree node to JSON. */
  struct JsonNodeVisitor : public TreeNode::Visitor {
    json result;
    const GroupNames *group_names = nullptr;
    void visit(const TreeCondition& node) override;
    void visit(const TreeResponse& node) override;
  };

  /** @brief Visitor that serializes a model (Tree or Forest) to JSON. */
  struct JsonModelVisitor : public Model::Visitor {
    json result;
    const GroupNames *group_names = nullptr;
    void visit(const Tree& tree) override;
    void visit(const Forest& forest) override;
  };

  /** @brief Map integer response codes to group name strings. */
  std::vector<std::string> to_labels(
    const types::ResponseVector&    predictions,
    const std::vector<std::string>& group_names);

  /** @name Serialization */
  ///@{
  json to_json(const Model& model);
  json to_json(const TreeNode& node);
  json to_json(const Tree& tree);
  json to_json(const BootstrapTree& tree);
  json to_json(const Forest& forest);
  json to_json(const stats::ConfusionMatrix& cm);
  json to_json(const VariableImportance& vi);
  json to_json(const types::FeatureMatrix& matrix);
  ///@}

  /** @name Labeled serialization (uses group names instead of integer codes) */
  ///@{
  json to_json(const Model& model, const GroupNames& group_names);
  json to_json(const TreeNode& node, const GroupNames& group_names);
  json to_json(const Tree& tree, const GroupNames& group_names);
  json to_json(const BootstrapTree& tree, const GroupNames& group_names);
  json to_json(const Forest& forest, const GroupNames& group_names);
  json to_json(const stats::ConfusionMatrix& cm, const GroupNames& group_names);
  ///@}

  /** @name Deserialization */
  ///@{

  /**
   * @brief Deserialize from a model block (integer labels only).
   *
   * For labeled JSON (string values/groups), use `j.get<Export<T>>()` instead.
   * Call via `serialization::from_json<T>(j)` or `j.get<T>()`.
   */
  template<typename T> T from_json(const json& j);

  template<> Tree from_json<Tree>(const json& j);
  template<> Forest from_json<Forest>(const json& j);
  template<> stats::ConfusionMatrix from_json<stats::ConfusionMatrix>(const json& j);
  template<> VariableImportance from_json<VariableImportance>(const json& j);
  ///@}

  /** @name Stream operators */
  ///@{
  std::ostream& operator<<(std::ostream& os, const TreeNode& node);
  std::ostream& operator<<(std::ostream& os, const TreeCondition& condition);
  std::ostream& operator<<(std::ostream& os, const TreeResponse& response);
  std::ostream& operator<<(std::ostream& os, const Tree& tree);
  std::ostream& operator<<(std::ostream& os, const Forest& forest);

  template<typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::vector<V> &vec) {
    json json_vector(vec);
    return ostream << json_vector.dump();
  }

  template<typename V, typename C1, typename C2>
  std::ostream& operator<<(std::ostream& ostream, const std::set<V, C1, C2> &set) {
    json json_set(set);
    return ostream << json_set.dump();
  }

  template<typename K, typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::map<K, V> &map) {
    json json_map(map);
    return ostream << json_map.dump();
  }

  template<typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::map<int, V> &map) {
    std::map<std::string, V> string_map;

    for (const auto& [key, val] : map) {
      string_map[std::to_string(key)] = val;
    }

    json json_map(string_map);
    return ostream << json_map.dump();
  }

  ///@}
}

// ADL serializer specializations — enables j.get<T>() for all types.
namespace nlohmann {
  template<> struct adl_serializer<ppforest2::Tree> {
    static ppforest2::Tree from_json(const json& j) {
      return ppforest2::serialization::from_json<ppforest2::Tree>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::Forest> {
    static ppforest2::Forest from_json(const json& j) {
      return ppforest2::serialization::from_json<ppforest2::Forest>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::stats::ConfusionMatrix> {
    static ppforest2::stats::ConfusionMatrix from_json(const json& j) {
      return ppforest2::serialization::from_json<ppforest2::stats::ConfusionMatrix>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::VariableImportance> {
    static ppforest2::VariableImportance from_json(const json& j) {
      return ppforest2::serialization::from_json<ppforest2::VariableImportance>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Tree>> {
    static ppforest2::serialization::Export<ppforest2::Tree> from_json(const json& j);
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Forest>> {
    static ppforest2::serialization::Export<ppforest2::Forest> from_json(const json& j);
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Model::Ptr>> {
    static ppforest2::serialization::Export<ppforest2::Model::Ptr> from_json(const json& j);
  };
}
