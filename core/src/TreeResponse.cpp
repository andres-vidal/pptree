#include "TreeResponse.hpp"

#include <memory>
#include <ostream>

using namespace models::types;

namespace models {
  TreeResponse::TreeResponse(Response value)
    : value(value) {
  }

  void TreeResponse::accept(TreeNodeVisitor& visitor) const {
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

  json TreeResponse::to_json() const {
    return json{ { "value", value } };
  }

  TreeNode::Ptr TreeResponse::clone() const {
    return std::make_unique<TreeResponse>(*this);
  }

  TreeResponse::Ptr TreeResponse::make(Response value) {
    return std::make_unique<TreeResponse>(value);
  }

  std::ostream& operator<<(std::ostream& ostream, const TreeResponse& response) {
    return ostream << response.to_json().dump(2, ' ', false);
  }
}
