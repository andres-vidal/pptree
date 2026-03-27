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
#include <ostream>

/**
 * @brief JSON serialization and deserialization for ppforest2 models.
 *
 * Uses nlohmann/json.  Provides to_json() for serializing trees,
 * forests, confusion matrices, and variable importance to JSON, and
 * *_from_json() for deserializing them back.  Also provides ostream
 * operators for convenient text output.
 *
 * @code
 *   // Serialize a forest to JSON and write to file:
 *   auto j = serialization::to_json(forest);
 *   io::json::write_file(j, "model.json");
 *
 *   // Read and deserialize:
 *   auto j2 = io::json::read_file("model.json");
 *   Forest restored = serialization::forest_from_json(j2);
 * @endcode
 */
namespace ppforest2::serialization {
  using json = nlohmann::json;

  /** @brief Group name vector for labeled serialization. */
  using GroupNames = std::vector<std::string>;

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
  Model::Ptr model_from_json(const json& j);
  TreeNode::Ptr node_from_json(const json& j);
  Tree tree_from_json(const json& j);
  Forest forest_from_json(const json& j);

  /** @name Labeled deserialization (maps string labels back to integer codes) */
  TreeNode::Ptr node_from_json(const json& j, const GroupNames& group_names);
  Tree tree_from_json(const json& j, const GroupNames& group_names);
  Forest forest_from_json(const json& j, const GroupNames& group_names);
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
}
