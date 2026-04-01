#pragma once

#include "models/DRStrategy.hpp"
#include "models/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::dr {
  /**
   * @brief No-op dimensionality reduction: selects all variables.
   *
   * Used with standard (non-random-forest) trees where all features
   * are available to the projection pursuit step at every node.
   */
  struct DRNoopStrategy : public DRStrategy {
    void to_json(nlohmann::json& j) const override;
    std::string display_name() const override { return "All variables"; }

    DRResult
    select(types::FeatureMatrix const& x, stats::GroupPartition const& group_spec, stats::RNG& rng) const override;

    static DRStrategy::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(DRStrategy, "noop")
  };

  /** @brief Factory function for a no-op DR strategy. */
  DRStrategy::Ptr noop();
}
