#pragma once

#include "models/strategies/partition/StepPartition.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::partition {
  /**
   * @brief Group-based partition: route all observations of a group to the same child.
   *
   * This is the default PPtree partition strategy. All observations
   * belonging to a group are sent to the same child node, determined
   * by the binary regrouping. The subgroups mapping is used to recover
   * the original class labels for each child.
   */
  struct ByGroup : public StepPartition {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By group"; }

    /**
     * @brief NodeContext-based interface: partition using binary_y subgroups.
     */
    Result split(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: partition from binary partition and group labels.
     */
    Result compute(stats::GroupPartition const& binary_y, types::Outcome group_0, types::Outcome group_1) const;

    static StepPartition::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(StepPartition, "by_group")
  };
}
