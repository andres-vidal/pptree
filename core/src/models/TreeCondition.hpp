#pragma once

#include "utils/Types.hpp"
#include "models/TreeNode.hpp"
#include "models/TrainingSpec.hpp"
#include "utils/Math.hpp"

namespace pptree {
  /** @brief Scalar threshold type for split decisions. */
  using Threshold = types::Feature;

  /**
   * @brief Internal split node in a projection pursuit tree.
   *
   * Projects an observation onto @c projector and compares the result
   * against @c threshold.  If the projected value is below the threshold
   * the observation goes to the @c lower child; otherwise to @c upper.
   */
  struct TreeCondition final : public TreeNode {
    using Ptr = std::unique_ptr<TreeCondition>;

    /** @brief Projection vector (p). Defines the linear combination of features. */
    pp::Projector projector;
    /** @brief Split threshold on the projected value. */
    Threshold threshold;
    /** @brief Child node for observations with projected value < threshold. */
    TreeNode::Ptr lower;
    /** @brief Child node for observations with projected value ≥ threshold. */
    TreeNode::Ptr upper;

    /** @brief Training specification used at this split (may be null). */
    TrainingSpec::Ptr training_spec = nullptr;
    /** @brief Set of class labels reachable from this node. */
    std::set<types::Response> classes;
    /** @brief Projection pursuit index value achieved at this split. */
    types::Feature pp_index_value = 0;

    TreeCondition(pp::Projector projector,
      Threshold                 threshold,
      TreeNode::Ptr             lower,
      TreeNode::Ptr             upper,
      TrainingSpec::Ptr         training_spec  = nullptr,
      std::set<types::Response> classes        = {},
      types::Feature            pp_index_value = 0);

    void accept(TreeNodeVisitor& visitor) const override;

    /** @brief Returns the response of the lower child. */
    types::Response response() const override;

    /**
     * @brief Route an observation through this split.
     *
     * Projects @p data onto the projector, compares against the
     * threshold, and delegates to the appropriate child node.
     *
     * @param data  Feature vector (p).
     * @return      Predicted class label from the reached leaf.
     */
    types::Response predict(const types::FeatureVector& data) const override;

    int class_count() const override {
      return static_cast<int>(classes.size());
    }

    std::set<types::Response> node_classes() const override {
      return classes;
    }

    bool equals(const TreeNode& other) const override;
    TreeNode::Ptr clone() const override;

    /** @brief Factory method that returns a unique_ptr to a new TreeCondition. */
    static Ptr make(pp::Projector projector,
      Threshold                   threshold,
      TreeNode::Ptr               lower,
      TreeNode::Ptr               upper,
      TrainingSpec::Ptr           training_spec  = nullptr,
      std::set<types::Response>   classes        = {},
      types::Feature              pp_index_value = 0);
  };
}
