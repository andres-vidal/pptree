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
     * @brief NodeContext-based interface: partition using binary_y subgroups.
     */
    Result split(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: partition from binary partition and group labels.
     */
    Result compute(stats::GroupPartition const& binary_y, types::GroupId group_0, types::GroupId group_1) const;

    static Grouping::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_label")
  };
}
