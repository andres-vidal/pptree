#pragma once

#include "Types.hpp"
#include "TreeNode.hpp"
#include "TreeCondition.hpp"
#include "TreeResponse.hpp"
#include "TrainingSpec.hpp"

#include <nlohmann/json.hpp>
#include <ostream>

namespace models {
  using json = nlohmann::json;

  struct Tree {
    static Tree train(
      TrainingSpec const&    training_spec,
      types::FeatureMatrix&  x,
      types::ResponseVector& y,
      stats::RNG&            rng);

    static Tree train(
      TrainingSpec const&          training_spec,
      types::FeatureMatrix&        x,
      stats::GroupPartition const& group_spec,
      stats::RNG&                  rng);

    TreeNode::Ptr root;
    TrainingSpec::Ptr training_spec;

    explicit Tree(TreeNode::Ptr root);
    Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    types::Response predict(const types::FeatureVector& data) const;
    types::ResponseVector predict(const types::FeatureMatrix& data) const;

    bool operator==(const Tree& other) const;
    bool operator!=(const Tree& other) const;

    json to_json() const;
    static Tree from_json(const json& j);
  };

  std::ostream& operator<<(std::ostream& ostream, const Tree& tree);
}
