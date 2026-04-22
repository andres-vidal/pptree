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
   * Writes `ctx.y_bin`.
   */
  struct Binarization : public Strategy<Binarization> {
    /** @brief Reduce a multiclass partition to binary and store in context. */
    virtual void regroup(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for regroup(). Skips if `ctx.aborted` is set. */
    void operator()(NodeContext& ctx, stats::RNG& rng) const;
  };

  /** @brief Factory function for largest-gap binarization. */
  Binarization::Ptr largest_gap();

  /** @brief Factory function for the Disabled (placeholder) binarizer. */
  Binarization::Ptr disabled();
}
