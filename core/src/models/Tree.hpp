#pragma once

#include "utils/Types.hpp"

#include "models/Model.hpp"
#include "models/TreeNode.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/TrainingSpec.hpp"


namespace pptree {
  struct Tree : public Model {
    static Tree train(
      TrainingSpec const&          training_spec,
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y,
      stats::RNG&                  rng);

    static Tree train(
      TrainingSpec const&          training_spec,
      const types::FeatureMatrix&  x,
      stats::GroupPartition const& group_spec,
      stats::RNG&                  rng);

    TreeNode::Ptr root;
    TrainingSpec::Ptr training_spec;

    explicit Tree(TreeNode::Ptr root);
    Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    void accept(ModelVisitor& visitor) const override;

    types::Response predict(const types::FeatureVector& data) const override;
    types::ResponseVector predict(const types::FeatureMatrix& data) const override;

    bool operator==(const Tree& other) const;
    bool operator!=(const Tree& other) const;
  };
}
