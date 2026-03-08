#pragma once
#include <algorithm>

#include "stats/GroupPartition.hpp"

#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "models/Projector.hpp"
#include "utils/Types.hpp"



namespace pptree::dr {
  struct DRSpec {
    const std::vector<int> selected_cols;
    const size_t original_size;

    DRSpec(const std::vector<int>& selected_cols, const size_t original_size) :
      selected_cols(selected_cols),
      original_size(original_size) {
    }

    pp::Projector expand(const pp::Projector& reduced_vector) const {
      invariant(reduced_vector.size() == selected_cols.size(), "Reduced vector size must match number of selected variables");

      pp::Projector full_vector = pp::Projector::Zero(original_size);

      for (size_t i = 0; i < selected_cols.size(); ++i) {
        full_vector(selected_cols[i]) = reduced_vector(i);
      }

      return full_vector;
    }
  };

  struct DRStrategy {
    using Ptr = std::unique_ptr<DRStrategy>;

    virtual ~DRStrategy()                 = default;
    virtual DRStrategy::Ptr clone() const = 0;

    virtual DRSpec select(
      types::FeatureMatrix const &  x,
      stats::GroupPartition const & group_spec,
      stats::RNG &                  rng) const = 0;

    DRSpec operator()(
      types::FeatureMatrix const &  x,
      stats::GroupPartition const & group_spec,
      stats::RNG &                  rng) const {
      return select(x, group_spec, rng);
    }
  };
}
