#pragma once

#include "utils/Types.hpp"

namespace ppforest2 {
  /**
   * @brief Grouped result of the variable importance measures.
   *
   * Produced by `Tree::variable_importance` (fills `projections` + `scale`)
   * and `Forest::variable_importance` (fills all four fields).
   *
   * Individual measures can be computed with `Tree::vi_projections`,
   * `Forest::vi_projections`, `Forest::vi_permuted`, and
   * `Forest::vi_weighted_projections`.
   */
  struct VariableImportance {
    /** @brief VI1 — per-variable permuted importance (forest only). */
    types::FeatureVector permuted;
    /** @brief VI2 — per-variable projection-coefficient importance. */
    types::FeatureVector projections;
    /** @brief VI3 — per-variable weighted-projection importance (forest only). */
    types::FeatureVector weighted_projections;
    /** @brief Per-variable σ used to rescale coefficients (columnwise sd). */
    types::FeatureVector scale;
  };
}
