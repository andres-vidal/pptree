#pragma once

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::grouping {
  /**
   * @brief Cutpoint-based grouping for regression trees.
   *
   * At each node, observations are split by the cutpoint in projected space,
   * then each child's observations are sorted by continuous_y and
   * median-split into 2 artificial groups for the next PDA projection.
   *
   * Requires the non-const ctx.x and ctx.y_vec on NodeContext for in-place
   * reordering of data within each node's range.
   *
   * - `init()`: wraps pre-computed GroupIdVector (median-split of sorted y)
   *   into a GroupPartition (same as ByLabel).
   * - `split()`: partitions observations by cutpoint, re-sorts each child
   *   by continuous_y, median-splits into 2 groups.
   */
  struct ByCutpoint : public Grouping {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By cutpoint"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Regression}; }

    // Bring the base-class `init(GroupIdVector)` convenience overload into
    // scope so it isn't hidden by the virtual override below.
    using Grouping::init;

    /**
     * @brief Median-split the sorted continuous response into 2 groups.
     *
     * The caller must pre-sort `y` ascending. For `n >= 2` the result is
     * `GroupPartition::two_groups(0, mid-1, mid, n-1)`; for `n == 1` a
     * single-group partition covering `[0, 0]` is returned.
     */
    stats::GroupPartition init(types::OutcomeVector const& y) const override;

    /**
     * @brief Split observations by cutpoint and re-cluster each child.
     *
     * Requires the non-const ctx.x and ctx.y_vec on the context.
     * 1. Partitions rows within the node's range by projected value vs cutpoint.
     * 2. Sorts each child's rows by continuous_y.
     * 3. Median-splits each child into 2 groups.
     */
    void split(NodeContext& ctx, types::GroupId lower, types::GroupId upper, stats::RNG& rng) const override;

    static Grouping::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_cutpoint")
  };

  /** @brief Factory function for cutpoint-based grouping. */
  Grouping::Ptr by_cutpoint();
}
