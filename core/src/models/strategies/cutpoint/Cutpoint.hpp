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
   * Writes `ctx.cutpoint`.
   */
  struct Cutpoint : public Strategy<Cutpoint> {
    /** @brief Compute the split cutpoint and store it in the context. */
    virtual void cutpoint(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for cutpoint(). Skips if `ctx.aborted` is set. */
    void operator()(NodeContext& ctx, stats::RNG& rng) const;
  };

  /** @brief Factory function for mean-of-means split cutpoint. */
  Cutpoint::Ptr mean_of_means();
}
