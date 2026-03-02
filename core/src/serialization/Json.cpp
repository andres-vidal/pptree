#include "serialization/Json.hpp"
#include "models/Projector.hpp"

#include <Eigen/Dense>

using namespace pptree::types;

namespace pptree::serialization {
  void JsonNodeVisitor::visit(const TreeCondition& node) {
    JsonNodeVisitor lower_visitor;
    node.lower->accept(lower_visitor);

    JsonNodeVisitor upper_visitor;
    node.upper->accept(upper_visitor);

    result = json{
      { "projector", node.projector },
      { "threshold", node.threshold },
      { "lower",     lower_visitor.result },
      { "upper",     upper_visitor.result },
    };
  }

  void JsonNodeVisitor::visit(const TreeResponse& node) {
    result = json{ { "value", node.value } };
  }

  json to_json(const TreeNode& node) {
    JsonNodeVisitor visitor;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(const Tree& tree) {
    return json{ { "root", to_json(*tree.root) } };
  }

  json to_json(const BootstrapTree& tree) {
    return to_json(static_cast<const Tree&>(tree));
  }

  json to_json(const Forest& forest) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (const auto& tree : forest.trees) {
      trees_json.push_back(to_json(*tree));
    }

    return json{ { "trees", trees_json } };
  }

  json to_json(const stats::ConfusionMatrix& cm) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.push_back(cm.values(i, col));
      }
      matrix_data.push_back(row);
    }
    j["matrix"] = matrix_data;

    std::vector<int> labels;
    for (const auto& [label, idx] : cm.label_index) {
      labels.push_back(label);
    }
    j["labels"] = labels;

    auto ce = cm.class_errors();
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
    j["class_errors"] = ce_vec;

    return j;
  }

  TreeNode::Ptr node_from_json(const json& j) {
    if (j.contains("value")) {
      return TreeResponse::make(j["value"].get<Response>());
    }

    const auto proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<const pp::Projector>(
      proj_vec.data(),
      static_cast<int>(proj_vec.size()));

    const Feature threshold = j["threshold"].get<Feature>();

    auto lower = node_from_json(j["lower"]);
    auto upper = node_from_json(j["upper"]);

    return TreeCondition::make(projector, threshold, std::move(lower), std::move(upper));
  }

  Tree tree_from_json(const json& j) {
    return Tree(node_from_json(j["root"]));
  }

  Forest forest_from_json(const json& j) {
    Forest forest;

    for (const auto& tree_json : j.at("trees")) {
      forest.add_tree(std::make_unique<BootstrapTree>(node_from_json(tree_json.at("root"))));
    }

    return forest;
  }

  std::ostream& operator<<(std::ostream& os, const TreeNode& node) {
    return os << to_json(node).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const TreeCondition& condition) {
    return os << to_json(static_cast<const TreeNode&>(condition)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const TreeResponse& response) {
    return os << to_json(static_cast<const TreeNode&>(response)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const Tree& tree) {
    return os << to_json(tree).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const Forest& forest) {
    return os << to_json(forest).dump(2, ' ', false);
  }
}
