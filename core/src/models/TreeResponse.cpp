#include "models/TreeResponse.hpp"

#include <memory>

using namespace ppforest2::types;

namespace ppforest2 {
  TreeResponse::TreeResponse(Response value) :
    value(value) {
  }

  void TreeResponse::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Response TreeResponse::response() const {
    return value;
  }

  Response TreeResponse::predict(const FeatureVector& data) const {
    return value;
  }

  bool TreeResponse::equals(const TreeNode& other) const {
    const auto *resp = dynamic_cast<const TreeResponse *>(&other);
    return resp && (value == resp->value);
  }

  TreeNode::Ptr TreeResponse::clone() const {
    return std::make_unique<TreeResponse>(*this);
  }

  TreeResponse::Ptr TreeResponse::make(Response value) {
    return std::make_unique<TreeResponse>(value);
  }
}
