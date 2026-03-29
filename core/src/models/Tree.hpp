#pragma once

#include "utils/Types.hpp"

#include "models/Model.hpp"
#include "models/TreeNode.hpp"


namespace ppforest2 {
  /**
   * @brief A single projection pursuit decision tree.
   *
   * Each internal node projects data onto a linear combination of
   * features and splits on the projected value.  Leaf nodes hold
   * group labels.
   *
   * @code
   *   TrainingSpec spec(pp::pda(0.0), dr::noop(), sr::mean_of_means());
   *   stats::RNG rng(0);
   *
   *   Tree tree = Tree::train(spec, x, y, rng);
   *   types::Response label = tree.predict(x.row(0));
   *   types::ResponseVector preds = tree.predict(x);
   * @endcode
   */
  struct Tree : public Model {
    /**
     * @brief Train a tree from a response vector.
     *
     * Constructs a GroupPartition from @p y and delegates to the
     * GroupPartition overload.
     *
     * @param training_spec  Training specification (strategy + DR).
     * @param x              Feature matrix (n × p).
     * @param y              Response vector (n).
     * @param rng            Random number generator.
     * @return               Trained tree.
     */
    static Tree train(
      TrainingSpec const&          training_spec,
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y,
      stats::RNG&                  rng);

    /**
     * @brief Train a tree from a group partition.
     *
     * Recursively splits the data by finding optimal projections
     * at each node until pure leaves are reached.
     *
     * @param training_spec  Training specification (strategy + DR).
     * @param x              Feature matrix (n × p).
     * @param group_spec     Group partition.
     * @param rng            Random number generator.
     * @return               Trained tree.
     */
    static Tree train(
      TrainingSpec const&          training_spec,
      const types::FeatureMatrix&  x,
      stats::GroupPartition const& group_spec,
      stats::RNG&                  rng);

    /** @brief Root node of the tree. */
    TreeNode::Ptr root;

    Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    void accept(Model::Visitor& visitor) const override;

    types::Response predict(const types::FeatureVector& data) const override;
    types::ResponseVector predict(const types::FeatureMatrix& data) const override;
    types::FeatureMatrix predict(const types::FeatureMatrix& data, Proportions) const override;

    bool operator==(const Tree& other) const;
    bool operator!=(const Tree& other) const;
  };
}
