#pragma once

#include "models/Projector.hpp"
#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Projection pursuit strategies.
 *
 * Contains the abstract ProjectionPursuit interface and concrete
 * implementations (e.g. PDA) that define how to evaluate
 * and optimise a projection index for separating groups.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::pp {
  /**
   * @brief Abstract strategy for projection pursuit optimization.
   *
   * Finds the optimal 1D projection for a given dataset and group
   * partition. Reads from NodeContext: x, var_selection, and the
   * active partition (y or binary->y). Writes: projector
   * (full-dimensional, expanded) and pp_index_value.
   */
  struct ProjectionPursuit : public Strategy<ProjectionPursuit> {
    /**
     * @brief Result of a projection pursuit optimization step.
     */
    struct Result {
      /** @brief Optimized projection vector. */
      ppforest2::pp::Projector projector;
      /** @brief Projection pursuit index value achieved. */
      types::Feature index_value = 0;
    };

    /**
     * @brief Find the optimal projection and store it in the context.
     *
     * Reads ctx.x, ctx.var_selection, and ctx.active_partition(). Computes the
     * optimal projector on the reduced feature space, expands it back
     * to full dimension via ctx.var_selection.expand(), and writes the result to
     * ctx.projector and ctx.pp_index_value.
     *
     * @param ctx  Node context (reads x, var_selection, active partition; writes projector, pp_index_value).
     * @param rng  Random number generator (unused by deterministic strategies).
     */
    virtual void optimize(NodeContext& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for optimize(). Skips if `ctx.aborted` is set. */
    void operator()(NodeContext& ctx, stats::RNG& rng) const;
  };

  /** @brief Factory function for a PDA projection pursuit strategy. */
  ProjectionPursuit::Ptr pda(float lambda);
}
