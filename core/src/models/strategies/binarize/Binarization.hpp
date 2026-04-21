#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Binarization strategies for multiclass-to-binary reduction.
 *
 * When a node has more than 2 groups, a binarization strategy reduces
 * it to a binary problem. The built-in LargestGap finds the largest
 * gap between sorted projected group means. Future strategies (e.g.
 * closest-pair from da Silva Extension I) can be plugged in.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::binarize {
  /**
   * @brief Abstract strategy for multiclass-to-binary reduction.
   *
   * Receives the node context with the first projector already set
   * (from find_projection on G groups). Projects data using
   * ctx.projector and produces a 2-group binary partition.
   *
   * Reads from NodeContext: x, y, projector.
   * Writes: binary_y, binary_0, binary_1.
   */
  struct Binarization : public Strategy<Binarization> {
    /**
     * @brief Result of a binarization step (for direct computation).
     */
    struct Result {
      /** @brief 2-group partition with subgroups mapping to original groups. */
      stats::GroupPartition binary_y;
      /** @brief Group label for binary group 0. */
      types::GroupId group_0;
      /** @brief Group label for binary group 1. */
      types::GroupId group_1;
    };

    /**
     * @brief Reduce a multiclass partition to binary and store in context.
     *
     * @param ctx  Node context (reads x, y, projector; writes binary_y, binary_0, binary_1).
     * @param rng  Random number generator (unused by deterministic strategies).
     */
    virtual void regroup(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for regroup(). */
    void operator()(NodeContext& ctx, stats::RNG& rng) const { regroup(ctx, rng); }
  };

  /** @brief Factory function for largest-gap binarization. */
  Binarization::Ptr largest_gap();

  /** @brief Factory function for the Disabled (placeholder) binarizer. */
  Binarization::Ptr disabled();
}
