#pragma once

#include "models/TreeNode.hpp"

namespace ppforest2 {
  /**
   * @brief Leaf node in a projection pursuit tree.
   *
   * Holds a single class label and always returns it as the prediction,
   * regardless of the input feature vector.
   */
  struct TreeResponse : public TreeNode {
    using Ptr = std::unique_ptr<TreeResponse>;

    /** @brief Class label stored at this leaf. */
    types::Response value;

    explicit TreeResponse(types::Response value);

    void accept(TreeNodeVisitor& visitor) const override;
    types::Response response() const override;

    /**
     * @brief Return the stored class label (ignores input).
     *
     * @param data  Feature vector (unused).
     * @return      The class label stored at this leaf.
     */
    types::Response predict(const types::FeatureVector& data) const override;

    bool is_leaf() const override {
      return true;
    }

    int class_count() const override {
      return 1;
    }

    std::set<types::Response> node_classes() const override {
      return { value };
    }

    bool equals(const TreeNode& other) const override;
    TreeNode::Ptr clone() const override;

    /** @brief Factory method that returns a unique_ptr to a new TreeResponse. */
    static Ptr make(types::Response value);
  };
}
