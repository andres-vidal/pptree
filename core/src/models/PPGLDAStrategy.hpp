#pragma once

#include "utils/Types.hpp"
#include "models/PPStrategy.hpp"

namespace pptree::pp {
  struct PPGLDAStrategy : public PPStrategy {
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

  PPStrategy::Ptr glda(float lambda);
}
