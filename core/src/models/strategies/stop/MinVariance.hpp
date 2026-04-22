#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the response variance is below a threshold.
   *
   * Useful for regression trees. A node becomes a leaf when the
   * variance of the continuous response among its observations
   * falls below the specified threshold.
   *
   * Requires NodeContext::continuous_y to be set.
   */
  struct MinVariance : public StopRule {
    types::Feature threshold;

    explicit MinVariance(types::Feature threshold);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Regression}; }

    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const override;

    static StopRule::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_variance")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_variance", "threshold")
  };

  /** @brief Factory function for minimum-variance stop rule. */
  StopRule::Ptr min_variance(types::Feature threshold);
}
