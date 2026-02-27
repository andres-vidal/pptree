#pragma once

#include "TreeNode.hpp"

namespace models {
  struct TreeResponse : public TreeNode {
    using Ptr = std::unique_ptr<TreeResponse>;

    types::Response value;

    explicit TreeResponse(types::Response value);

    void accept(TreeNodeVisitor& visitor) const override;
    types::Response response() const override;
    types::Response predict(const types::FeatureVector& data) const override;

    bool equals(const TreeNode& other) const override;
    json to_json() const override;

    TreeNode::Ptr clone() const override;

    static Ptr make(types::Response value);
  };

  std::ostream& operator<<(std::ostream& ostream, const TreeResponse& response);
}
