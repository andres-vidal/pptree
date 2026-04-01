#pragma once

#include "models/Projector.hpp"
#include "models/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "utils/Types.hpp"

#include <algorithm>
#include <vector>

/**
 * @brief Dimensionality reduction strategies for variable selection.
 *
 * Contains the abstract DRStrategy interface and concrete
 * implementations that select a subset of variables before projection
 * pursuit optimisation.  DRNoopStrategy uses all variables (single
 * trees); DRUniformStrategy samples uniformly at random (forests).
 *
 * New strategies must implement the pure virtual methods including
 * to_json() for serialization support.
 */
namespace ppforest2::dr {
  /**
   * @brief Result of a dimensionality reduction step.
   *
   * Records which columns were selected and allows expanding a
   * reduced-dimension projector back to the original feature space.
   */
  struct DRResult {
    /** @brief Indices of the selected columns in the original matrix. */
    std::vector<int> const selected_cols;
    /** @brief Number of columns in the original (unreduced) matrix. */
    size_t const original_size;

    DRResult(std::vector<int> const& selected_cols, size_t const original_size)
        : selected_cols(selected_cols)
        , original_size(original_size) {}

    /**
     * @brief Expand a reduced-dimension projector to the original space.
     *
     * Places each element of @p reduced_vector at the position of
     * the corresponding selected column; all other positions are zero.
     *
     * @param reduced_vector  Projector in the reduced space (q).
     * @return                Projector in the original space (p), zero-padded.
     */
    pp::Projector expand(pp::Projector const& reduced_vector) const {
      invariant(
          reduced_vector.size() == selected_cols.size(), "Reduced vector size must match number of selected variables"
      );

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
   *
   * Implementations must also provide to_json() for serialization.
   */
  struct DRStrategy : public Strategy<DRStrategy> {
    /**
     * @brief Select a subset of variables.
     *
     * @param x           Feature matrix (n × p).
     * @param group_spec  Group partition.
     * @param rng         Random number generator.
     * @return            DRResult recording which columns were selected.
     */
    virtual DRResult
    select(types::FeatureMatrix const& x, stats::GroupPartition const& group_spec, stats::RNG& rng) const = 0;

    /**
     * @brief Convenience operator: equivalent to select().
     */
    DRResult operator()(types::FeatureMatrix const& x, stats::GroupPartition const& group_spec, stats::RNG& rng) const {
      return select(x, group_spec, rng);
    }
  };
}
