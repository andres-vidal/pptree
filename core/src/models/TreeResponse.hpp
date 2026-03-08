#pragma once

#include "models/TreeNode.hpp"

namespace pptree {
  struct TreeResponse : public TreeNode {
    using Ptr = std::unique_ptr<TreeResponse>;

    types::Response value;

    explicit TreeResponse(types::Response value);

    void accept(TreeNodeVisitor& visitor) const override;
    types::Response response() const override;
    types::Response predict(const types::FeatureVector& data) const override;

    bool equals(const TreeNode& other) const override;

    TreeNode::Ptr clone() const override;

    static Ptr make(types::Response value);
  };
}
