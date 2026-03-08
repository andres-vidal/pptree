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

    int class_count() const override { return 1; }
    std::set<types::Response> node_classes() const override { return { value }; }

    bool equals(const TreeNode& other) const override;

    TreeNode::Ptr clone() const override;

    static Ptr make(types::Response value);
  };
}
