#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

/**
 * @brief Cutpoint strategies for computing decision cutpoints.
 *
 * Contains the abstract Cutpoint interface and concrete implementations
 * that determine the split cutpoint in projected space. The built-in
 * MeanOfMeans uses the midpoint of the two group means.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::cutpoint {
  /**
   * @brief Abstract strategy for computing the split cutpoint.
   *
   * Given the current node context (with projector and active partition),
   * determines the cutpoint value that separates the two groups in the
   * projected space.
   *
   * Reads from NodeContext: x, projector, active_partition().
   * Writes: cutpoint.
   */
  struct Cutpoint : public Strategy<Cutpoint> {
    /**
     * @brief Compute the split cutpoint and store it in the context.
     *
     * @param ctx  Node context (reads x, projector, active partition; writes cutpoint).
     * @param rng  Random number generator (unused by deterministic strategies).
     */
    virtual void cutpoint(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for cutpoint(). */
    void operator()(NodeContext& ctx, stats::RNG& rng) const { cutpoint(ctx, rng); }
  };

  /** @brief Factory function for mean-of-means split cutpoint. */
  Cutpoint::Ptr mean_of_means();
}
