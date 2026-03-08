#pragma once

#include <memory>
#include <set>

#include "utils/Types.hpp"
#include "models/TreeNodeVisitor.hpp"

namespace pptree {
  /**
   * @brief Abstract base class for nodes in a projection pursuit tree.
   *
   * A tree is a recursive structure of nodes: internal split nodes
   * (TreeCondition) that project and threshold, and leaf nodes
   * (TreeResponse) that hold a class label.
   */
  struct TreeNode {
    using Ptr = std::unique_ptr<TreeNode>;

    virtual ~TreeNode() = default;

    /** @brief Accept a tree node visitor (double dispatch). */
    virtual void accept(TreeNodeVisitor &visitor) const = 0;

    /**
     * @brief Predict the class label for a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Predicted class label.
     */
    virtual types::Response predict(const types::FeatureVector &data) const = 0;

    /** @brief The class label at this node (leaf value or majority class). */
    virtual types::Response response() const = 0;

    /**
     * @brief Number of distinct classes reachable from this node.
     */
    virtual int class_count() const = 0;

    /**
     * @brief Sorted set of class labels reachable from this node.
     */
    virtual std::set<types::Response> node_classes() const = 0;

    /** @brief Structural equality comparison (value-based). */
    virtual bool equals(const TreeNode &other) const = 0;

    /** @brief Deep copy of this node and its subtree. */
    virtual Ptr clone() const = 0;

    bool operator==(const TreeNode &other) const;
    bool operator!=(const TreeNode &other) const;
  };
}
