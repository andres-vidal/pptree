#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Grouping strategies that manage group partitions throughout training.
 *
 * The Grouping strategy owns the full lifecycle of GroupPartitions:
 * initial construction from training labels (`init`) and per-node
 * child splitting (`split`).
 *
 * For classification, ByLabel constructs from sorted labels and routes
 * groups to children via the binary mapping.  For regression (future),
 * ByCutpoint quantile-slices the continuous response and re-clusters
 * children at each node.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::grouping {
  /**
   * @brief Abstract strategy for managing group partitions.
   *
   * Provides two operations:
   * - `init()`: create the root GroupPartition from training labels.
   * - `split()`: route observations to child nodes after a cutpoint.
   */
  struct Grouping : public Strategy<Grouping> {
    /**
     * @brief Result of a split step: two child group partitions.
     */
    struct Result {
      /** @brief Group partition for the lower (left) child. */
      stats::GroupPartition lower;
      /** @brief Group partition for the upper (right) child. */
      stats::GroupPartition upper;
    };

    /**
     * @brief Create the initial GroupPartition from training response.
     *
     * Called once before training begins. `y` is carried as `OutcomeVector`
     * for both modes — integer class labels (classification) or continuous
     * response (regression). Rows must be pre-sorted by the caller so that
     * equal values (classification) or ascending response (regression) form
     * contiguous blocks.
     *
     * - `ByLabel::init`: casts `y` to integer labels and builds a contiguous
     *   block partition over the unique class labels.
     * - `ByCutpoint::init`: median-splits the sorted response into two
     *   artificial groups for the initial regression projection.
     *
     * @param y  Response vector (n), must be sorted.
     * @return   The root GroupPartition.
     */
    virtual stats::GroupPartition init(types::OutcomeVector const& y) const = 0;

    /** @brief Convenience overload for callers with a `GroupIdVector`. */
    stats::GroupPartition init(types::GroupIdVector const& y) const {
      return init(types::OutcomeVector(y.cast<types::Outcome>()));
    }

    /**
     * @brief Split observations into two child partitions.
     *
     * @param ctx  Node context (reads binary_y, binary_0, binary_1, and subgroups).
     * @param rng  Random number generator (unused by deterministic strategies).
     * @return     Result with lower and upper child partitions.
     */
    virtual Result split(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for split(). */
    Result operator()(NodeContext& ctx, stats::RNG& rng) const { return split(ctx, rng); }
  };

  /** @brief Factory function for label-based grouping. */
  Grouping::Ptr by_label();

  /** @brief Factory function for cutpoint-based grouping (regression). */
  Grouping::Ptr by_cutpoint();
}
