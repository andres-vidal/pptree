#pragma once
#include <algorithm>

#include "stats/GroupPartition.hpp"

#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "models/Projector.hpp"
#include "utils/Types.hpp"



namespace pptree::dr {
  /**
   * @brief Result of a dimensionality reduction step.
   *
   * Records which columns were selected and allows expanding a
   * reduced-dimension projector back to the original feature space.
   */
  struct DRSpec {
    /** @brief Indices of the selected columns in the original matrix. */
    const std::vector<int> selected_cols;
    /** @brief Number of columns in the original (unreduced) matrix. */
    const size_t original_size;

    DRSpec(const std::vector<int>& selected_cols, const size_t original_size) :
      selected_cols(selected_cols),
      original_size(original_size) {
    }

    /**
     * @brief Expand a reduced-dimension projector to the original space.
     *
     * Places each element of @p reduced_vector at the position of
     * the corresponding selected column; all other positions are zero.
     *
     * @param reduced_vector  Projector in the reduced space (q).
     * @return                Projector in the original space (p), zero-padded.
     */
    pp::Projector expand(const pp::Projector& reduced_vector) const {
      invariant(reduced_vector.size() == selected_cols.size(), "Reduced vector size must match number of selected variables");

      pp::Projector full_vector = pp::Projector::Zero(original_size);

      for (size_t i = 0; i < selected_cols.size(); ++i) {
        full_vector(selected_cols[i]) = reduced_vector(i);
      }

      return full_vector;
    }
  };

  /**
   * @brief Abstract strategy for dimensionality reduction.
   *
   * Before projection pursuit optimization, a DR strategy selects a
   * subset of variables (columns) to work with.  This reduces the
   * cost of the PP step and introduces randomness in forests.
   */
  struct DRStrategy {
    using Ptr = std::unique_ptr<DRStrategy>;

    virtual ~DRStrategy()                 = default;
    virtual DRStrategy::Ptr clone() const = 0;

    /**
     * @brief Select a subset of variables.
     *
     * @param x           Feature matrix (n × p).
     * @param group_spec  Group partition.
     * @param rng         Random number generator.
     * @return            DRSpec recording which columns were selected.
     */
    virtual DRSpec select(
      types::FeatureMatrix const &  x,
      stats::GroupPartition const & group_spec,
      stats::RNG &                  rng) const = 0;

    /**
     * @brief Convenience operator: equivalent to select().
     */
    DRSpec operator()(
      types::FeatureMatrix const &  x,
      stats::GroupPartition const & group_spec,
      stats::RNG &                  rng) const {
      return select(x, group_spec, rng);
    }
  };
}
