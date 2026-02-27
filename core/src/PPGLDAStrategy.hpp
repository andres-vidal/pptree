#pragma once

#include "Types.hpp"
#include "PPStrategy.hpp"

namespace models::pp::strategy {
  struct PPGLDAStrategy : public PPStrategy {
    const float lambda;

    explicit PPGLDAStrategy(float lambda);

    PPStrategy::Ptr clone() const override;

    types::Feature index(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec,
      const Projector&             projector) const override;

    Projector optimize(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec) const override;

    static PPStrategy::Ptr make(float lambda);
  };

  PPStrategy::Ptr glda(float lambda);
}
