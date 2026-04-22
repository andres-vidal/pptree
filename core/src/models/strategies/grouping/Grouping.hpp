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
   * `init()` builds the root partition once before training; `split()`
   * writes `ctx.lower_y_part` and `ctx.upper_y_part` per node.
   */
  struct Grouping : public Strategy<Grouping> {
    /**
     * @brief Create the initial GroupPartition from the training response.
     *
     * Caller must pre-sort rows so equal values (classification) or ascending
     * response (regression) form contiguous blocks.
     */
    virtual stats::GroupPartition init(types::OutcomeVector const& y) const = 0;

    /** @brief Convenience overload for callers with a `GroupIdVector`. */
    stats::GroupPartition init(types::GroupIdVector const& y) const {
      return init(types::OutcomeVector(y.cast<types::Outcome>()));
    }

    /** @brief Split observations into two child partitions; writes ctx.lower_y_part / upper_y_part. */
    virtual void split(NodeContext& ctx, types::GroupId lower, types::GroupId upper, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for split(). Skips if `ctx.aborted` is set. */
    void operator()(NodeContext& ctx, types::GroupId lower, types::GroupId upper, stats::RNG& rng) const;
  };

  /** @brief Factory function for label-based grouping. */
  Grouping::Ptr by_label();

  /** @brief Factory function for cutpoint-based grouping (regression). */
  Grouping::Ptr by_cutpoint();
}
