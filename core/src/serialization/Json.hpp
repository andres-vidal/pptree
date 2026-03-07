#pragma once

#include "models/TreeNodeVisitor.hpp"
#include "models/TreeNode.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/ModelVisitor.hpp"
#include "models/Tree.hpp"
#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>
#include <ostream>

namespace pptree::serialization {
  using json = nlohmann::json;

  struct JsonNodeVisitor : public TreeNodeVisitor {
    json result;
    void visit(const TreeCondition& node) override;
    void visit(const TreeResponse& node) override;
  };

  struct JsonModelVisitor : public ModelVisitor {
    json result;
    void visit(const Tree& tree) override;
    void visit(const Forest& forest) override;
  };

  json to_json(const Model& model);
  json to_json(const TreeNode& node);
  json to_json(const Tree& tree);
  json to_json(const BootstrapTree& tree);
  json to_json(const Forest& forest);
  json to_json(const stats::ConfusionMatrix& cm);
  json to_json(const VariableImportance& vi);

  Model::Ptr model_from_json(const json& j);
  TreeNode::Ptr node_from_json(const json& j);
  Tree tree_from_json(const json& j);
  Forest forest_from_json(const json& j);

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
