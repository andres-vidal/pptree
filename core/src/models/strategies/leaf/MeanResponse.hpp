#pragma once

#include "models/strategies/leaf/LeafStrategy.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::leaf {
  /**
   * @brief Leaf creation by mean response value.
   *
   * Returns a TreeLeaf whose value is the mean of the continuous
   * response for observations in this node. Used for regression trees.
   *
   * Requires `NodeContext::y_vec` to be set (populated by the training
   * pipeline for regression; null for classification).
   */
  struct MeanResponse : public LeafStrategy {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Mean response"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Regression}; }

    /**
     * @brief Create a mean-response leaf from the node's observations.
     */
    TreeNode::Ptr create_leaf(NodeContext const& ctx, stats::RNG& rng) const override;

    static LeafStrategy::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(LeafStrategy, "mean_response")
  };
}
