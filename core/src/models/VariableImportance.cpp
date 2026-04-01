#include "models/VariableImportance.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/VIVisitor.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

using namespace ppforest2::types;

namespace ppforest2 {
  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  static ResponseVector gather_labels(ResponseVector const& y, std::vector<int> const& idx) {
    ResponseVector labels(static_cast<int>(idx.size()));

    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
      labels(i) = y(idx[i]);
    }

    return labels;
  }

  // -------------------------------------------------------------------------
  // VI1 — Permuted importance
  // -------------------------------------------------------------------------
  FeatureVector
  variable_importance_permuted(Forest const& forest, FeatureMatrix const& x, ResponseVector const& y, int seed) {
    int const n_vars  = static_cast<int>(x.cols());
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(forest.trees.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (int k = 0; k < B; ++k) {
      BootstrapTree const& tree = *forest.trees[k];

      if (tree.degenerate) {
        continue;
      }

      std::vector<int> oob_idx = tree.oob_indices(n_total);

      if (oob_idx.empty()) {
        valid_trees++;
        continue;
      }

      ResponseVector oob_labels    = gather_labels(y, oob_idx);
      ResponseVector baseline_pred = tree.predict_oob(x, oob_idx);

      float const baseline_acc = stats::accuracy(baseline_pred, oob_labels);

      stats::RNG rng(static_cast<unsigned>(seed) ^ static_cast<unsigned>(k));
      int const n_oob = static_cast<int>(oob_idx.size());
      stats::Uniform uniform(0, n_oob - 1);

      // Copy OOB rows once — only column j changes per iteration.
      FeatureMatrix perm_x(n_oob, n_vars);

      for (int i = 0; i < n_oob; ++i) {
        perm_x.row(i) = x.row(oob_idx[i]);
      }

      ResponseVector perm_pred(n_oob);

      for (int j = 0; j < n_vars; ++j) {
        // Save and permute column j.
        FeatureVector col_saved    = perm_x.col(j);
        std::vector<int> row_order = uniform.distinct(n_oob, rng);

        for (int i = 0; i < n_oob; ++i) {
          perm_x(i, j) = col_saved(row_order[i]);
        }

        // Predict on permuted data.
        for (int i = 0; i < n_oob; ++i) {
          perm_pred(i) = tree.predict(static_cast<FeatureVector>(perm_x.row(i)));
        }

        float const perm_acc = stats::accuracy(perm_pred, oob_labels);
        importance(j) += baseline_acc - perm_acc;

        // Restore column j for next iteration.
        perm_x.col(j) = col_saved;
      }

      valid_trees++;
    }

    // Average over non-degenerate trees.
    if (valid_trees > 0) {
      importance /= static_cast<Feature>(valid_trees);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI2 — Projections importance (single tree)
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_projections(Tree const& tree, int n_vars, FeatureVector const* scale) {
    FeatureVector importance = FeatureVector::Zero(n_vars);

    VIVisitor visitor(n_vars, scale);
    tree.root->accept(visitor);

    for (int j = 0; j < n_vars; ++j) {
      importance(j) = static_cast<Feature>(visitor.vi2_contributions[static_cast<std::size_t>(j)]);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI2 — Projections importance (forest, averaged over trees)
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_projections(Forest const& forest, int n_vars, FeatureVector const* scale) {
    int const B = static_cast<int>(forest.trees.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (int k = 0; k < B; ++k) {
      BootstrapTree const& tree = *forest.trees[k];

      if (tree.degenerate) {
        continue;
      }

      VIVisitor visitor(n_vars, scale);
      tree.root->accept(visitor);

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += static_cast<Feature>(visitor.vi2_contributions[static_cast<std::size_t>(j)]);
      }

      valid_trees++;
    }

    if (valid_trees > 0) {
      importance /= static_cast<Feature>(valid_trees);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI3 — Weighted projections importance
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_weighted_projections(
      Forest const& forest, FeatureMatrix const& x, ResponseVector const& y, FeatureVector const* scale
  ) {
    int const n_vars  = static_cast<int>(x.cols());
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(forest.trees.size());

    // Count G = number of unique groups.
    std::set<Response> group_set;

    for (int i = 0; i < y.size(); ++i) {
      group_set.insert(y(i));
    }

    int const G = static_cast<int>(group_set.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (int k = 0; k < B; ++k) {
      BootstrapTree const& tree = *forest.trees[k];

      if (tree.degenerate) {
        continue;
      }

      std::vector<int> oob_idx = tree.oob_indices(n_total);

      Feature e_k = Feature(0);

      if (!oob_idx.empty()) {
        ResponseVector oob_labels = gather_labels(y, oob_idx);
        ResponseVector oob_preds  = tree.predict_oob(x, oob_idx);
        e_k                       = static_cast<Feature>(stats::error_rate(oob_preds, oob_labels));
      }

      VIVisitor visitor(n_vars, scale);
      tree.root->accept(visitor);

      Feature const weight = Feature(1) - e_k;

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += weight * static_cast<Feature>(visitor.vi3_contributions[static_cast<std::size_t>(j)]);
      }

      valid_trees++;
    }

    Feature const denom = static_cast<Feature>(valid_trees) * static_cast<Feature>(G - 1);

    if (denom > Feature(0)) {
      importance /= denom;
    }

    return importance;
  }
}
