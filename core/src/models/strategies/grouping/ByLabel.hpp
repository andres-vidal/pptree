#pragma once

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::grouping {
  /**
   * @brief Label-based grouping: create partitions from sorted class labels.
   *
   * This is the default classification grouping strategy.
   *
   * - `init()`: wraps sorted integer labels into a GroupPartition.
   * - `split()`: routes all observations of a group to the same child
   *   based on the binary regrouping. Uses the subgroups mapping to
   *   recover original class labels for each child.
   */
  struct ByLabel : public Grouping {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By label"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    // Bring the base-class `init(GroupIdVector)` convenience overload into
    // scope so it isn't hidden by the virtual override below.
    using Grouping::init;

    /**
     * @brief Create a GroupPartition from sorted integer-valued labels.
     *
     * Casts the float-typed `y` to `GroupIdVector` internally.
     */
    stats::GroupPartition init(types::OutcomeVector const& y) const override;

    /**
     * @brief NodeContext-based interface: partition using `active_partition()`.
     *
     * Writes `ctx.lower_y_part` and `ctx.upper_y_part`.
     */
    void split(NodeContext& ctx, types::GroupId lower, types::GroupId upper, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: partition from group partition and lower/upper labels.
     *
     * Returns a `std::pair<lower, upper>` of child partitions. For testing only —
     * production code goes through `split()` and reads results from the context.
     */
    std::pair<stats::GroupPartition, stats::GroupPartition>
    compute(stats::GroupPartition const& y_part, types::GroupId lower, types::GroupId upper) const;

    static Grouping::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_label")
  };
}
