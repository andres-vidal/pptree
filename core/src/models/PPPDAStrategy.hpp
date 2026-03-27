#pragma once

#include "utils/Types.hpp"
#include "models/PPStrategy.hpp"

namespace ppforest2::pp {
  /**
   * @brief Penalized Discriminant Analysis projection pursuit strategy.
   *
   * Optimizes a linear discriminant projection using a penalized
   * between-group / within-group variance ratio.  The @c lambda
   * parameter controls the penalty strength in the LDA index.
   */
  struct PPPDAStrategy : public PPStrategy {
    /** @brief Penalty parameter for the LDA index (0 = standard LDA). */
    const float lambda;

    explicit PPPDAStrategy(float lambda);

    PPStrategy::Ptr clone() const override;

    types::Feature index(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec,
      const Projector&             projector) const override;

    PPResult optimize(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec) const override;

    static PPStrategy::Ptr make(float lambda);
  };

  /**
   * @brief Factory function for a PDA projection pursuit strategy.
   *
   * @param lambda  Penalty parameter.
   * @return        Owned pointer to a PPPDAStrategy.
   */
  PPStrategy::Ptr pda(float lambda);
}
