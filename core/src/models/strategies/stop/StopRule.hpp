#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

/**
 * @brief Stop rule strategies that determine when to create leaf nodes.
 *
 * Controls tree growth by deciding when a node should stop splitting
 * and become a leaf. The built-in PureNode stops when only one group
 * remains. Future strategies may add max-depth or min-samples rules.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::stop {
  /**
   * @brief Abstract strategy for tree stopping rules.
   *
   * Called at each node before any projection or splitting is attempted.
   * If should_stop returns true, the node becomes a leaf.
   *
   * Reads from NodeContext: y, depth.
   */
  struct StopRule : public Strategy<StopRule> {
    /**
     * @brief Determine whether to stop growing at this node.
     *
     * @param ctx  Node context (reads y, depth).
     * @param rng  Random number generator (unused by deterministic rules).
     * @return     True if the node should become a leaf.
     */
    virtual bool should_stop(NodeContext const& ctx, stats::RNG& rng) const = 0;

    /** @brief Callable shorthand for should_stop(). */
    bool operator()(NodeContext const& ctx, stats::RNG& rng) const { return should_stop(ctx, rng); }
  };

  /** @brief Factory function for pure-node stop rule. */
  StopRule::Ptr pure_node();
}
