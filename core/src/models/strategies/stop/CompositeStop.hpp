#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

#include <vector>

namespace ppforest2::stop {
  /**
   * @brief Composite stop rule that fires when any child rule fires (logical OR).
   *
   * Useful for combining multiple stopping criteria, e.g.:
   * @code
   *   stop::any({stop::min_size(5), stop::min_variance(0.01)})
   * @endcode
   */
  struct CompositeStop : public StopRule {
    std::vector<StopRule::Ptr> rules;

    explicit CompositeStop(std::vector<StopRule::Ptr> rules);

    nlohmann::json to_json() const override;
    std::string display_name() const override;

    /**
     * @brief Supported modes = intersection of child rules' supported modes.
     *
     * A composite stop rule is only usable in a mode where *every* child
     * rule is usable. E.g. `any(pure_node, min_size)` supports only
     * classification (pure_node restricts it).
     */
    std::set<types::Mode> supported_modes() const override;

    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const override;

    static StopRule::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(StopRule, "any")
  };

  /** @brief Factory function for composite stop rule (logical OR). */
  StopRule::Ptr any(std::vector<StopRule::Ptr> rules);
}
