#pragma once

#include "utils/Types.hpp"
#include "models/Tree.hpp"

#include <vector>

namespace ppforest2 {
  struct BootstrapTree : public Tree {
    using Tree::Tree;
    using Ptr = std::unique_ptr<BootstrapTree>;

    /**
     * @brief Train a bootstrap tree.
     *
     * Samples rows from @p x according to @p group_spec, trains a tree
     * on the sampled data, and stores the sample indices for OOB queries.
     *
     * @param training_spec  Training specification (shared).
     * @param x             Feature matrix (n × p).
     * @param group_spec    Group partition for stratified sampling.
     * @param rng            Random number generator.
     * @return               Trained bootstrap tree.
     */
    static Ptr train(
        TrainingSpec::Ptr const& training_spec,
        types::FeatureMatrix const& x,
        stats::GroupPartition const& group_spec,
        stats::RNG& rng
    );

    std::vector<int> sample_indices;

    BootstrapTree(TreeNode::Ptr root, TrainingSpec::Ptr spec, std::vector<int> samp);

    /**
     * @brief Indices of observations not in the bootstrap sample.
     *
     * @param n_total  Total number of observations in the training set.
     * @return         Sorted vector of out-of-bag row indices.
     */
    std::vector<int> oob_indices(int n_total) const;

    /**
     * @brief Predict for a subset of rows (e.g. OOB indices).
     *
     * Typically used with oob_indices() to obtain predictions for
     * out-of-bag observations only.  The returned vector has the same
     * size as @p row_idx; element @c i is the prediction for row
     * @c row_idx[i].
     *
     * @param x         Feature matrix (n × p).
     * @param row_idx   Row indices to predict.
     * @return          Predictions for each row in row_idx.
     */
    types::OutcomeVector predict_oob(types::FeatureMatrix const& x, std::vector<int> const& row_idx) const;
  };
}
