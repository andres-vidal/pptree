#pragma once

#include "utils/Types.hpp"
#include "models/PPStrategy.hpp"

namespace pptree::pp {
  /**
   * @brief Generalized LDA projection pursuit strategy.
   *
   * Optimizes a linear discriminant projection using a penalized
   * between-class / within-class variance ratio.  The @c lambda
   * parameter controls the penalty strength in the LDA index.
   */
  struct PPGLDAStrategy : public PPStrategy {
    /** @brief Penalty parameter for the LDA index (0 = standard LDA). */
    const float lambda;

    explicit PPGLDAStrategy(float lambda);

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
   * @brief Factory function for a GLDA projection pursuit strategy.
   *
   * @param lambda  Penalty parameter.
   * @return        Owned pointer to a PPGLDAStrategy.
   */
  PPStrategy::Ptr glda(float lambda);
}
