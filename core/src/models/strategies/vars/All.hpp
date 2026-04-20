#pragma once

#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::vars {
  /**
   * @brief Selects all variables (no variable selection).
   *
   * Used with standard (non-random-forest) trees where all features
   * are available to the projection pursuit step at every node.
   */
  struct All : public VariableSelection {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "All variables"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    /**
     * @brief NodeContext-based interface: select all variables and write to ctx.var_selection.
     */
    void select(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: return all column indices.
     */
    Result compute(types::FeatureMatrix const& x) const;

    static VariableSelection::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(VariableSelection, "all")
  };
}
