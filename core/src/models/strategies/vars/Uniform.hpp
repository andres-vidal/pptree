#pragma once

#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

namespace ppforest2::vars {
  /**
   * @brief Uniform random variable selection.
   *
   * Selects @c n_vars variables uniformly at random (without
   * replacement) from the full set of features. Used in random
   * forests to introduce diversity between trees.
   */
  struct Uniform : public VariableSelection {
    explicit Uniform(int n_vars);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Uniform random"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    /**
     * @brief NodeContext-based interface: select variables and write to ctx.var_selection.
     */
    void select(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: select variables from the given matrix.
     */
    Result compute(types::FeatureMatrix const& x, stats::RNG& rng) const;

    static VariableSelection::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(VariableSelection, "uniform")
    PPFOREST2_REGISTER_PRIMARY_PARAM("uniform", "count")

  private:
    /** @brief Number of variables to select at each split. */
    int const n_vars;
  };
}
