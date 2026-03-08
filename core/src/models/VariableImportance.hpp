#pragma once

#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "utils/Types.hpp"

namespace pptree {
  /**
   * @brief Grouped result of the three variable importance measures.
   *
   * For single trees only @c projections and @c scale are populated;
   * @c permuted and @c weighted_projections remain empty (size 0).
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
    const Forest&                forest,
    const types::FeatureMatrix&  x,
    const types::ResponseVector& y,
    int                          seed = 0);

  /**
   * @brief VI2 — Projections importance (forest).
   *
   * At every split node s with G_s classes, accumulates |a_j| / G_s for
   * each variable j.  When @p scale is provided each |a_j| is first
   * multiplied by σ_j so that coefficients are comparable across
   * variables with different units.  Results are averaged over all trees.
   *
   * @param forest  Trained random forest.
   * @param n_vars  Number of predictor variables (p).
   * @param scale   Optional per-variable σ vector (size p).
   * @return        FeatureVector of size p with per-variable importance.
   */
  types::FeatureVector variable_importance_projections(
    const Forest&              forest,
    int                        n_vars,
    const types::FeatureVector *scale = nullptr);

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
  types::FeatureVector variable_importance_projections(
    const Tree&                tree,
    int                        n_vars,
    const types::FeatureVector *scale = nullptr);

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
    const Forest&                forest,
    const types::FeatureMatrix&  x,
    const types::ResponseVector& y,
    const types::FeatureVector   *scale = nullptr);
}
