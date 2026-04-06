#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Partition strategies that route observations to child nodes.
 *
 * Controls how observations are split between the left and right
 * children after the split cutpoint is determined. The built-in
 * ByGroup routes all observations of a group to the same child.
 * Future strategies (e.g. observation-level splitting from da Silva
 * Extension II) can split individual observations across children.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::partition {
  /**
   * @brief Abstract strategy for partitioning observations into children.
   *
   * Given the node context with binary partition, projector, and
   * cutpoint, determines which observations go to each child node.
   *
   * Reads from NodeContext: binary_y (with subgroups), binary_0, binary_1.
   */
  struct StepPartition : public Strategy<StepPartition> {
    /**
     * @brief Result of a partition step: two child group partitions.
     */
    struct Result {
      /** @brief Group partition for the lower (left) child. */
      stats::GroupPartition lower;
      /** @brief Group partition for the upper (right) child. */
      stats::GroupPartition upper;
    };

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

  /** @brief Factory function for group-based partition. */
  StepPartition::Ptr by_group();
}
