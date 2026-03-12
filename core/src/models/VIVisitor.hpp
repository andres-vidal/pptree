#pragma once

#include "models/TreeNodeVisitor.hpp"
#include "models/TreeCondition.hpp"
#include "utils/Types.hpp"

#include <vector>

namespace ppforest2 {
  /**
   * @brief Visitor that accumulates per-variable contributions for VI2 and VI3.
   *
   * Traverses a single tree and, at each split node s with G_s classes and
   * PP index I_s, accumulates:
   *
   *   vi2_contributions[j] += |a_j| / G_s
   *   vi3_contributions[j] += I_s * |a_j|
   *
   * When @p scale is provided, each |a_j| is multiplied by σ_j so that
   * coefficients are comparable across variables with different units.
   *
   * The caller is responsible for forest-level aggregation and normalization
   * described in Da Silva et al. (2021).
   */
  struct VIVisitor : public TreeNodeVisitor {
    /** @brief VI2 contributions per variable (size p). */
    std::vector<types::Feature> vi2_contributions;

    /** @brief VI3 contributions per variable (size p). */
    std::vector<types::Feature> vi3_contributions;

    /** @brief Optional per-variable σ vector (size p); scale[j] = σ_j. */
    const types::FeatureVector *scale = nullptr;

    /**
     * @brief Construct a visitor for a tree with @p n_vars variables.
     *
     * @param n_vars  Number of predictor variables (p).
     * @param scale   Optional per-variable σ vector (size p).
     */
    explicit VIVisitor(int n_vars, const types::FeatureVector *scale = nullptr)  :
      vi2_contributions(static_cast<std::size_t>(n_vars), 0.0f),
      vi3_contributions(static_cast<std::size_t>(n_vars), 0.0f),
      scale(scale) {
    }

    /**
     * @brief Visit a split node; accumulate contributions and recurse.
     *
     * @param node  Split node with projector and pp_index_value.
     */
    void visit(const TreeCondition &node) override {
      const int n_vars         = static_cast<int>(vi2_contributions.size());
      const int G_s            = static_cast<int>(node.classes.size());
      const types::Feature I_s = node.pp_index_value;

      for (int j = 0; j < n_vars && j < node.projector.size(); ++j) {
        types::Feature coeff = std::abs(node.projector(j));

        if (scale != nullptr && j < scale->size()) {
          coeff *= (*scale)(j);
        }

        if (G_s > 0) {
          vi2_contributions[static_cast<std::size_t>(j)] += coeff / static_cast<types::Feature>(G_s); // <- not std::abs(coeff)
        }

        vi3_contributions[static_cast<std::size_t>(j)] += I_s * coeff;
      }

      node.lower->accept(*this);
      node.upper->accept(*this);
    }

    /**
     * @brief Visit a leaf node; no contribution to VI.
     *
     * Leaf nodes hold only a class label and do not contribute to variable
     * importance.
     */
    void visit(const TreeResponse &) override {
    }
  };
}
