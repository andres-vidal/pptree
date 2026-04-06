#pragma once

#include "models/strategies/leaf/LeafStrategy.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::leaf {
  /**
   * @brief Leaf creation by majority vote.
   *
   * Returns a TreeLeaf whose label is the most frequent class
   * in the node's group partition. When there is only one group,
   * it is returned directly. On ties, the numerically smallest
   * group label wins (deterministic).
   */
  struct MajorityVote : public LeafStrategy {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Majority vote"; }

    /**
     * @brief Create a majority-vote leaf from the node's group partition.
     */
    TreeNode::Ptr create_leaf(NodeContext const& ctx, stats::RNG& rng) const override;

    static LeafStrategy::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(LeafStrategy, "majority_vote")
  };
}
