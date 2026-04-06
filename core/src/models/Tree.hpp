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
   *   Tree tree = Tree::train(spec, x, y);
   *   types::Outcome label = tree.predict(x.row(0));
   *   types::OutcomeVector preds = tree.predict(x);
   * @endcode
   */
  struct Tree : public Model {
    /**
     * @brief Train a tree from a response vector.
     *
     * Creates an RNG from training_spec.seed and trains the tree.
     *
     * @param training_spec  Training specification (strategy + DR).
     * @param x              Feature matrix (n × p).
     * @param y              Outcome vector (n).
     * @return               Trained tree.
     */
    static Tree train(TrainingSpec const& training_spec, types::FeatureMatrix const& x, types::OutcomeVector const& y);

    /**
     * @brief Train a tree from a response vector with an external RNG.
     *
     * Used internally by BootstrapTree and Forest, which manage
     * their own RNG streams.
     *
     * @param training_spec  Training specification (strategy + DR).
     * @param x              Feature matrix (n × p).
     * @param y              Outcome vector (n).
     * @param rng            Random number generator.
     * @return               Trained tree.
     */
    static Tree train(
        TrainingSpec const& training_spec, types::FeatureMatrix const& x, types::OutcomeVector const& y, stats::RNG& rng
    );

    /**
     * @brief Train a tree from a group partition.
     *
     * Creates an RNG from training_spec.seed and trains the tree.
     */
    static Tree
    train(TrainingSpec const& training_spec, types::FeatureMatrix const& x, stats::GroupPartition const& group_spec);

    /**
     * @brief Train a tree from a group partition with an external RNG.
     *
     * Used internally by BootstrapTree and Forest.
     */
    static Tree train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix const& x,
        stats::GroupPartition const& group_spec,
        stats::RNG& rng
    );

    /** @brief Root node of the tree. */
    TreeNode::Ptr root;

    Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    void accept(Model::Visitor& visitor) const override;

    types::Outcome predict(types::FeatureVector const& data) const override;
    types::OutcomeVector predict(types::FeatureMatrix const& data) const override;
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    bool operator==(Tree const& other) const;
    bool operator!=(Tree const& other) const;
  };
}
