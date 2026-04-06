#pragma once

#include "models/TreeNode.hpp"

namespace ppforest2 {
  /**
   * @brief Leaf node in a projection pursuit tree.
   *
   * Holds a single class label and always returns it as the prediction,
   * regardless of the input feature vector.
   */
  struct TreeLeaf : public TreeNode {
    using Ptr = std::unique_ptr<TreeLeaf>;

    /** @brief Class label stored at this leaf. */
    types::Outcome value;

    explicit TreeLeaf(types::Outcome value);

    void accept(TreeNode::Visitor& visitor) const override;
    types::Outcome response() const override;

    /**
     * @brief Return the stored class label (ignores input).
     *
     * @param data  Feature vector (unused).
     * @return      The class label stored at this leaf.
     */
    types::Outcome predict(types::FeatureVector const& data) const override;

    bool is_leaf() const override { return true; }

    int group_count() const override { return 1; }

    std::set<types::Outcome> node_groups() const override { return {value}; }

    bool equals(TreeNode const& other) const override;
    TreeNode::Ptr clone() const override;

    /** @brief Factory method that returns a unique_ptr to a new TreeLeaf. */
    static Ptr make(types::Outcome value);
  };
}
