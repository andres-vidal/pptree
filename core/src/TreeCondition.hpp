#pragma once

#include "Types.hpp"
#include "TreeNode.hpp"
#include "TrainingSpec.hpp"
#include "Math.hpp"

namespace models {
  struct TreeCondition final : public TreeNode {
    using Ptr = std::unique_ptr<TreeCondition>;

    pp::Projector projector;
    Threshold threshold;
    TreeNode::Ptr lower;
    TreeNode::Ptr upper;

    TrainingSpec::Ptr training_spec = nullptr;
    std::set<types::Response> classes; // <- not const, so move/copy works cleanly

    TreeCondition(pp::Projector projector,
      Threshold                 threshold,
      TreeNode::Ptr             lower,
      TreeNode::Ptr             upper,
      TrainingSpec::Ptr         training_spec = nullptr,
      std::set<types::Response> classes       = {});

    void accept(TreeNodeVisitor& visitor) const override;
    types::Response response() const override;
    types::Response predict(const types::FeatureVector& data) const override;

    bool equals(const TreeNode& other) const override;
    json to_json() const override;
    TreeNode::Ptr clone() const override;

    static Ptr make(pp::Projector projector,
      Threshold                   threshold,
      TreeNode::Ptr               lower,
      TreeNode::Ptr               upper,
      TrainingSpec::Ptr           training_spec = nullptr,
      std::set<types::Response>   classes       = {});
  };

  std::ostream& operator<<(std::ostream& ostream, const TreeCondition& condition);
}
