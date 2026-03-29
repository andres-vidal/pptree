#pragma once

#include "models/DRStrategy.hpp"
#include "models/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::dr {
  /**
   * @brief Uniform random dimensionality reduction.
   *
   * Selects @c n_vars variables uniformly at random (without
   * replacement) from the full set of features.  Used in random
   * forests to introduce diversity between trees.
   */
  struct DRUniformStrategy : public DRStrategy {
    explicit DRUniformStrategy(int n_vars);

    void to_json(nlohmann::json& j) const override;
    std::string display_name() const override {
      return "Uniform random";
    }

    DRResult select(
      types::FeatureMatrix const&  x,
      stats::GroupPartition const& group_spec,
      stats::RNG&                  rng) const override;

    static DRStrategy::Ptr from_json(const nlohmann::json& j);

    PPFOREST2_REGISTER_STRATEGY(DRStrategy, "uniform")

    private:
      /** @brief Number of variables to select at each split. */
      const int n_vars;
  };

  /** @brief Factory function for a uniform DR strategy. */
  DRStrategy::Ptr uniform(int n_vars);
}
