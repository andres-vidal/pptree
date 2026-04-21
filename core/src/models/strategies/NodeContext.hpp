#pragma once

#include "models/Projector.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

#include <optional>

using Projector = ppforest2::pp::Projector;

namespace ppforest2 {
  /**
   * @brief Mutable context accumulating intermediate results during node training.
   *
   * Starts with node-level data (x, y, depth) and accumulates results
   * as each strategy in the training pipeline executes. Each strategy
   * reads what it needs and writes its results back.
   */
  struct NodeContext {
    /** @brief Full feature matrix (immutable reference to training data). */
    types::FeatureMatrix const& x;
    /** @brief Original G-group partition for this node. */
    stats::GroupPartition const& y;
    /** @brief Depth of this node in the tree. */
    int depth;

    /**
     * @brief Optional raw response vector (for regression).
     *
     * When set, points at the full-length `y` vector whose values match the
     * current `x` layout. Regression strategies (`MeanResponse` leaf,
     * `MinVariance` stop) read it to compute arithmetic on the response.
     * Null for classification (classification strategies use `y`'s
     * GroupPartition exclusively).
     */
    types::OutcomeVector const* y_vec = nullptr;

    /**
     * @brief Mutable data pointers for in-place reordering (regression only).
     *
     * ByCutpoint needs to physically reorder data within each node's range
     * to maintain sorted-by-y groups for child partitions. These are null
     * for classification (no reordering needed).
     */
    types::FeatureMatrix* mutable_x = nullptr;
    types::OutcomeVector* mutable_y_vec = nullptr;

    /** @brief Set by select_vars: variable selection result. */
    vars::VariableSelection::Result var_selection;

    /** @brief Set by find_projection: optimized projector (full dimension, expanded). */
    Projector projector;
    /** @brief Set by find_projection: projection pursuit index value achieved. */
    types::Feature pp_index_value = 0;

    /** @brief Set by regroup (multiclass -> binary): 2-group partition with subgroups. */
    std::optional<stats::GroupPartition> binary_y;
    /** @brief Set by regroup: group label assigned to binary group 0. */
    types::GroupId binary_0 = -1;
    /** @brief Set by regroup: group label assigned to binary group 1. */
    types::GroupId binary_1 = -1;

    /** @brief Set by find_cutpoint: split cutpoint in projected space. */
    types::Feature cutpoint = 0;

    NodeContext(types::FeatureMatrix const& x, stats::GroupPartition const& y, int depth)
        : x(x)
        , y(y)
        , depth(depth)
        , projector(Projector::Zero(x.cols())) {}

    /**
     * @brief Return the active group partition.
     *
     * After binarization, returns the binary partition (2 groups).
     * Before binarization (or for 2-group nodes), returns the original partition.
     */
    stats::GroupPartition const& active_partition() const { return binary_y.has_value() ? binary_y.value() : y; }
  };
}
