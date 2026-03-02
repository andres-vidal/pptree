#pragma once

#include "models/Projector.hpp"
#include "stats/GroupPartition.hpp"
#include <set>
#include <vector>
#include <memory>

namespace pptree::pp {
  struct PPStrategy {
    using Ptr = std::unique_ptr<PPStrategy>;

    virtual ~PPStrategy()     = default;
    virtual Ptr clone() const = 0;

    virtual types::Feature index(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec,
      const Projector&             projector) const = 0;

    virtual Projector optimize(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec) const = 0;

    Projector operator()(const types::FeatureMatrix &x, const stats::GroupPartition& group_spec) const {
      return optimize(x, group_spec);
    }
  };
}
