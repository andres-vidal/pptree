#pragma once

#include "utils/Types.hpp"
#include "utils/JsonValidation.hpp"
#include "models/PPStrategy.hpp"
#include "models/Strategy.hpp"

namespace ppforest2::pp {
  /**
   * @brief Penalized Discriminant Analysis projection pursuit strategy.
   *
   * Optimizes a linear discriminant projection using a penalized
   * between-group / within-group variance ratio.  The @c lambda
   * parameter controls the penalty strength in the LDA index.
   */
  struct PPPDAStrategy : public PPStrategy {
    explicit PPPDAStrategy(float lambda);

    void to_json(nlohmann::json& j) const override;
    std::string display_name() const override {
      return lambda == 0 ? "LDA" : "PDA";
    }

    types::Feature index(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec,
      const Projector&             projector) const override;

    PPResult optimize(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec) const override;

    static PPStrategy::Ptr from_json(const nlohmann::json& j);

    PPFOREST2_REGISTER_STRATEGY(PPStrategy, "pda")

    private:
      /** @brief Penalty parameter for the LDA index (0 = standard LDA). */
      const float lambda;
  };

  /** @brief Factory function for a PDA projection pursuit strategy. */
  PPStrategy::Ptr pda(float lambda);
}
