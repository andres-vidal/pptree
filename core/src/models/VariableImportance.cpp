#include "models/VariableImportance.hpp"
#include "models/VIVisitor.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

using namespace pptree::types;

namespace pptree {
  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  static ResponseVector gather_labels(
    const ResponseVector&   y,
    const std::vector<int>& idx) {
    ResponseVector labels(static_cast<int>(idx.size()));

    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
      labels(i) = y(idx[i]);
    }

    return labels;
  }

  // -------------------------------------------------------------------------
  // VI1 — Permuted importance
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_permuted(
    const Forest&         forest,
    const FeatureMatrix&  x,
    const ResponseVector& y,
    int                   seed) {
    const int n_vars  = static_cast<int>(x.cols());
    const int n_total = static_cast<int>(x.rows());
    const int B       = static_cast<int>(forest.trees.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);

    for (int k = 0; k < B; ++k) {
      const BootstrapTree& tree = *forest.trees[k];
      std::vector<int> oob_idx  = tree.oob_indices(n_total);

      if (oob_idx.empty()) {
        continue;
      }

      ResponseVector oob_labels    = gather_labels(y, oob_idx);
      ResponseVector baseline_pred = tree.predict_oob(x, oob_idx);

      const float baseline_acc = stats::accuracy(baseline_pred, oob_labels);

      stats::RNG rng(static_cast<unsigned>(seed) ^ static_cast<unsigned>(k));
      const int n_oob = static_cast<int>(oob_idx.size());
      stats::Uniform uniform(0, n_oob - 1);

      for (int j = 0; j < n_vars; ++j) {
        // Build a permuted copy of the OOB rows with column j shuffled.
        FeatureMatrix perm_x(n_oob, n_vars);

        for (int i = 0; i < n_oob; ++i) {
          perm_x.row(i) = x.row(oob_idx[i]);
        }

        // Permute column j in-place.
        std::vector<int> row_order = uniform.distinct(n_oob, rng);

        FeatureVector col_copy = perm_x.col(j);

        for (int i = 0; i < n_oob; ++i) {
          perm_x(i, j) = col_copy(row_order[i]);
        }

        // Predict on permuted data.
        ResponseVector perm_pred(n_oob);

        for (int i = 0; i < n_oob; ++i) {
          perm_pred(i) = tree.predict(static_cast<FeatureVector>(perm_x.row(i)));
        }

        const float perm_acc = stats::accuracy(perm_pred, oob_labels);
        importance(j) += baseline_acc - perm_acc;
      }
    }

    // Average over trees.
    if (B > 0) {
      importance /= static_cast<Feature>(B);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI2 — Projections importance (single tree)
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_projections(
    const Tree&         tree,
    int                 n_vars,
    const FeatureVector *scale) {
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
  FeatureVector variable_importance_projections(
    const Forest&       forest,
    int                 n_vars,
    const FeatureVector *scale) {
    const int B = static_cast<int>(forest.trees.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);

    for (int k = 0; k < B; ++k) {
      const BootstrapTree& tree = *forest.trees[k];

      VIVisitor visitor(n_vars, scale);
      tree.root->accept(visitor);

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += static_cast<Feature>(visitor.vi2_contributions[static_cast<std::size_t>(j)]);
      }
    }

    if (B > 0) {
      importance /= static_cast<Feature>(B);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI3 — Weighted projections importance
  // -------------------------------------------------------------------------
  FeatureVector variable_importance_weighted_projections(
    const Forest&         forest,
    const FeatureMatrix&  x,
    const ResponseVector& y,
    const FeatureVector   *scale) {
    const int n_vars  = static_cast<int>(x.cols());
    const int n_total = static_cast<int>(x.rows());
    const int B       = static_cast<int>(forest.trees.size());

    // Count G = number of unique classes.
    std::set<Response> class_set;

    for (int i = 0; i < y.size(); ++i) {
      class_set.insert(y(i));
    }

    const int G = static_cast<int>(class_set.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);

    for (int k = 0; k < B; ++k) {
      const BootstrapTree& tree = *forest.trees[k];
      std::vector<int> oob_idx  = tree.oob_indices(n_total);

      Feature e_k = Feature(0);

      if (!oob_idx.empty()) {
        ResponseVector oob_labels = gather_labels(y, oob_idx);
        ResponseVector oob_preds  = tree.predict_oob(x, oob_idx);
        e_k = static_cast<Feature>(stats::error_rate(oob_preds, oob_labels));
      }

      VIVisitor visitor(n_vars, scale);
      tree.root->accept(visitor);

      const Feature weight = Feature(1) - e_k;

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += weight * static_cast<Feature>(visitor.vi3_contributions[static_cast<std::size_t>(j)]);
      }
    }

    const Feature denom = static_cast<Feature>(B) * static_cast<Feature>(G - 1);

    if (denom > Feature(0)) {
      importance /= denom;
    }

    return importance;
  }
}
