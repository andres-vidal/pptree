#include "models/TreeLeaf.hpp"

#include <memory>

using namespace ppforest2::types;

namespace ppforest2 {
  TreeLeaf::TreeLeaf(Outcome value)
      : value(value) {}

  void TreeLeaf::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Outcome TreeLeaf::response() const {
    return value;
  }

  Outcome TreeLeaf::predict(FeatureVector const& data) const {
    return value;
  }

  bool TreeLeaf::equals(TreeNode const& other) const {
    auto const* resp = dynamic_cast<TreeLeaf const*>(&other);
    return resp && (value == resp->value);
  }

  TreeNode::Ptr TreeLeaf::clone() const {
    return std::make_unique<TreeLeaf>(*this);
  }

  TreeLeaf::Ptr TreeLeaf::make(Outcome value) {
    return std::make_unique<TreeLeaf>(value);
  }
}
