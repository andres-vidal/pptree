#pragma once

#include "models/Projector.hpp"
#include "models/strategies/binarize/Binarization.hpp"
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
    /**
     * @brief Full feature matrix.
     *
     * Non-const because regression's `ByCutpoint` grouping strategy
     * reorders rows in place within each node's range. Classification
     * strategies only read — there's no mode-specific subtype, so the
     * const promise lives at the strategy level, not in the context.
     */
    types::FeatureMatrix& x;
    /** @brief Original G-group partition for this node. */
    stats::GroupPartition const& y;
    /** @brief Depth of this node in the tree. */
    int depth;

    /**
     * @brief Raw response vector whose row order matches `x`.
     *
     * Regression strategies (`MeanResponse` leaf, `MinVariance` stop,
     * `ByCutpoint` grouping) read or reorder it; classification
     * strategies ignore it. Non-const for the same reason as `x`.
     * Always set — no mode-conditional logic at the call site.
     */
    types::OutcomeVector* y_vec;

    /** @brief Set by select_vars: variable selection result. `std::nullopt` before select_vars runs. */
    std::optional<vars::VariableSelection::Result> var_selection;

    /** @brief Set by find_projection: optimized projector (full dimension, expanded). `std::nullopt` before find_projection runs. */
    std::optional<Projector> projector;
    /** @brief Set by find_projection: projection pursuit index value achieved. `std::nullopt` before find_projection runs. */
    std::optional<types::Feature> pp_index_value;

    /**
     * @brief Set by regroup (multiclass → binary).
     *
     * Bundles the 2-group partition and the two binary group labels.
     * `std::nullopt` before regroup fires (or when the node was
     * already binary and skipped binarization).
     */
    std::optional<binarize::Binarization::Result> binary;

    /** @brief Set by find_cutpoint: split cutpoint in projected space. `std::nullopt` before find_cutpoint runs. */
    std::optional<types::Feature> cutpoint;

    NodeContext(types::FeatureMatrix& x, stats::GroupPartition const& y, types::OutcomeVector& y_vec, int depth)
        : x(x)
        , y(y)
        , depth(depth)
        , y_vec(&y_vec) {}

    /**
     * @brief Return the active group partition.
     *
     * After binarization, returns the binary partition (2 groups).
     * Before binarization (or for 2-group nodes), returns the original partition.
     */
    stats::GroupPartition const& active_partition() const { return binary.has_value() ? binary->y : y; }
  };
}
