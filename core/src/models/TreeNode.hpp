#pragma once

#include <memory>
#include <set>

#include "utils/Types.hpp"

namespace ppforest2 {
  struct TreeBranch;
  struct TreeLeaf;

  /**
   * @brief Abstract base class for nodes in a projection pursuit tree.
   *
   * A tree is a recursive structure of nodes: internal split nodes
   * (TreeBranch) that project and split, and leaf nodes
   * (TreeLeaf) that hold a group label.
   */
  struct TreeNode {
    using Ptr = std::unique_ptr<TreeNode>;

    /**
     * @brief Visitor interface for tree node dispatch.
     *
     * Implements the visitor pattern to distinguish between internal
     * split nodes (TreeBranch) and leaf nodes (TreeLeaf).
     */
    struct Visitor {
      virtual void visit(TreeBranch const& condition) = 0;
      virtual void visit(TreeLeaf const& response)    = 0;
    };

    /** @brief Whether this node (or any descendant) had a degenerate split. */
    bool degenerate = false;

    virtual ~TreeNode() = default;

    /** @brief Accept a tree node visitor (double dispatch). */
    virtual void accept(Visitor& visitor) const = 0;

    /**
     * @brief Predict the group label for a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Predicted group label.
     */
    virtual types::Outcome predict(types::FeatureVector const& data) const = 0;

    /** @brief The group label at this node (leaf value or majority group). */
    virtual types::Outcome response() const = 0;

    /**
     * @brief Number of distinct groups reachable from this node.
     */
    virtual int group_count() const = 0;

    /**
     * @brief Sorted set of group labels reachable from this node.
     */
    virtual std::set<types::GroupId> node_groups() const = 0;

    /** @brief Whether this node is a leaf (TreeLeaf). */
    virtual bool is_leaf() const = 0;

    /** @brief Structural equality comparison (value-based). */
    virtual bool equals(TreeNode const& other) const = 0;

    /** @brief Deep copy of this node and its subtree. */
    virtual Ptr clone() const = 0;

    bool operator==(TreeNode const& other) const;
    bool operator!=(TreeNode const& other) const;
  };
}
