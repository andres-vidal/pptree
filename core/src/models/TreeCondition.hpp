#pragma once

#include "utils/Types.hpp"
#include "models/TreeNode.hpp"
#include "models/TrainingSpec.hpp"
#include "utils/Math.hpp"

namespace pptree {
  using Threshold = types::Feature;

  struct TreeCondition final : public TreeNode {
    using Ptr = std::unique_ptr<TreeCondition>;

    pp::Projector projector;
    Threshold threshold;
    TreeNode::Ptr lower;
    TreeNode::Ptr upper;

    TrainingSpec::Ptr training_spec = nullptr;
    std::set<types::Response> classes; // <- not const, so move/copy works cleanly
    types::Feature pp_index_value = 0;

    TreeCondition(pp::Projector projector,
      Threshold                 threshold,
      TreeNode::Ptr             lower,
      TreeNode::Ptr             upper,
      TrainingSpec::Ptr         training_spec  = nullptr,
      std::set<types::Response> classes        = {},
      types::Feature            pp_index_value = 0);

    void accept(TreeNodeVisitor& visitor) const override;
    types::Response response() const override;
    types::Response predict(const types::FeatureVector& data) const override;

    bool equals(const TreeNode& other) const override;
    TreeNode::Ptr clone() const override;

    static Ptr make(pp::Projector projector,
      Threshold                   threshold,
      TreeNode::Ptr               lower,
      TreeNode::Ptr               upper,
      TrainingSpec::Ptr           training_spec  = nullptr,
      std::set<types::Response>   classes        = {},
      types::Feature              pp_index_value = 0);
  };
}
