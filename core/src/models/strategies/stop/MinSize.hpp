#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the node has fewer than a minimum number of observations.
   *
   * Useful for regression trees where pure-node stopping doesn't apply.
   * A node becomes a leaf when the total number of observations across
   * all its groups is less than the specified minimum.
   */
  struct MinSize : public StopRule {
    int min_size;

    explicit MinSize(int min_size);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const override;

    static StopRule::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_size")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_size", "min_size")
  };

  /** @brief Factory function for minimum-size stop rule. */
  StopRule::Ptr min_size(int n);
}
