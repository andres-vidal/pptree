
#include "TreeNode.hpp"
#include "TreeResponse.hpp"
#include "TreeCondition.hpp"

#include <ostream>
#include <Eigen/Dense>

using namespace models::types;

namespace models {
  bool TreeNode::operator==(const TreeNode& other) const {
    return this->equals(other);
  }

  bool TreeNode::operator!=(const TreeNode& other) const {
    return !this->equals(other);
  }

  std::ostream& operator<<(std::ostream& ostream, const TreeNode& node) {
    return ostream << node.to_json().dump(2, ' ', false);
  }

  TreeNode::Ptr node_from_json(const json& j) {
    if (j.contains("value")) {
      return TreeResponse::make(j["value"].get<Response>());
    }

    const auto proj_vec     = j["projector"].get<std::vector<Feature> >();
    pp::Projector projector =
      Eigen::Map<const pp::Projector >(proj_vec.data(), static_cast<int>(proj_vec.size()));

    const Feature threshold = j["threshold"].get<Feature>();
    auto lower              = node_from_json(j["lower"]);
    auto upper              = node_from_json(j["upper"]);

    return TreeCondition::make(projector, threshold, std::move(lower), std::move(upper));
  }
}
