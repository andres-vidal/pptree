#pragma once

#include "utils/Types.hpp"

namespace ppforest2 {
  struct Tree;
  struct Forest;
  /**
   * @brief Grouped result of the three variable importance measures.
   *
   * For single trees only @c projections and @c scale are populated;
   * @c permuted and @c weighted_projections remain empty (size 0).
   *
   * @code
   *   // Forest — all three measures:
   *   auto vi1 = variable_importance_permuted(forest, x, y, seed: 0);
   *   auto vi2 = variable_importance_projections(forest, x.cols());
   *   auto vi3 = variable_importance_weighted_projections(forest, x, y);
   *
   *   // Single tree — projection-based only:
   *   auto vi = variable_importance_projections(tree, x.cols());
   * @endcode
   *
   * @see variable_importance_permuted, variable_importance_projections,
   *      variable_importance_weighted_projections
   */
  struct VariableImportance {
    types::FeatureVector permuted;
    types::FeatureVector projections;
    types::FeatureVector weighted_projections;
    types::FeatureVector scale;
  };

  /**
   * @brief VI1 — Permuted importance.
   *
   * For each tree, measures the drop in OOB accuracy when each variable
   * is randomly permuted among the OOB observations.  Results are
   * averaged over all trees.
   *
   * @param forest  Trained random forest.
   * @param x       Training feature matrix (n × p).
   * @param y       Training response vector (n).
   * @param seed    RNG seed for the permutations.
   * @return        FeatureVector of size p with per-variable importance.
   */
  types::FeatureVector variable_importance_permuted(
      Forest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed = 0
  );

  /**
   * @brief VI2 — Projections importance (forest).
   *
   * At every split node s with G_s groups, accumulates |a_j| / G_s for
   * each variable j.  When @p scale is provided each |a_j| is first
   * multiplied by σ_j so that coefficients are comparable across
   * variables with different units.  Results are averaged over all trees.
   *
   * @param forest  Trained random forest.
   * @param n_vars  Number of predictor variables (p).
   * @param scale   Optional per-variable σ vector (size p).
   * @return        FeatureVector of size p with per-variable importance.
   */
  types::FeatureVector
  variable_importance_projections(Forest const& forest, int n_vars, types::FeatureVector const* scale = nullptr);

  /**
   * @brief VI2 — Projections importance (single tree).
   *
   * Same as the forest overload but for a single tree (no averaging).
   *
   * @param tree    Trained tree.
   * @param n_vars  Number of predictor variables (p).
   * @param scale   Optional per-variable σ vector (size p).
   * @return        FeatureVector of size p with per-variable importance.
   */
  types::FeatureVector
  variable_importance_projections(Tree const& tree, int n_vars, types::FeatureVector const* scale = nullptr);

  /**
   * @brief VI3 — Weighted projections importance.
   *
   * Each tree's contribution is weighted by (1 − e_k), where e_k is its
   * OOB error rate, and each split node contributes I_s × |a_j|.  The
   * result is normalised by B × (G − 1).
   *
   * @param forest  Trained random forest.
   * @param x       Training feature matrix (n × p).
   * @param y       Training response vector (n).
   * @param scale   Optional per-variable σ vector (size p).
   * @return        FeatureVector of size p with per-variable importance.
   */
  types::FeatureVector variable_importance_weighted_projections(
      Forest const& forest,
      types::FeatureMatrix const& x,
      types::OutcomeVector const& y,
      types::FeatureVector const* scale = nullptr
  );
}
