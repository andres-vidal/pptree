#pragma once

#include "utils/Types.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>

namespace ppforest2 {
  /**
   * @brief Bootstrap-aggregated model wrapper.
   *
   * Pairs a concrete model with the row indices of the bootstrap sample it
   * was trained on, so out-of-bag queries can recover the complementary
   * observations. Template over `M` so the bagging abstraction is
   * orthogonal to the model being aggregated — today `M = Tree`, but
   * nothing in this class is tree-specific, and a future `Bagged<GBTree>`
   * or `Bagged<LinearModel>` would reuse the exact same wrapper.
   *
   * `M` must provide:
   *   - `predict(types::FeatureVector const&)` returning a scalar outcome
   *   - `predict(types::FeatureMatrix const&)` returning an OutcomeVector
   *   - `bool degenerate` (field or method) for the forest retry logic
   *   - `operator==(M const&)` for structural equality round-trips
   *
   * Lives in a header because it's a template; the implementations are
   * mechanical one-liners that profit from being inlined.
   */
  template<typename M> struct Bagged {
    using Ptr = std::unique_ptr<Bagged<M>>;

    /** @brief The bootstrap-trained model. */
    std::unique_ptr<M> model;
    /** @brief Row indices (into the original training set) used to train `model`. */
    std::vector<int> sample_indices;

    Bagged(std::unique_ptr<M> model, std::vector<int> sample_indices)
        : model(std::move(model))
        , sample_indices(std::move(sample_indices)) {}

    /** @brief Delegate single-row prediction to the wrapped model. */
    types::Outcome predict(types::FeatureVector const& x) const { return this->model->predict(x); }

    /** @brief Delegate batch prediction to the wrapped model. */
    types::OutcomeVector predict(types::FeatureMatrix const& x) const { return this->model->predict(x); }

    /**
     * @brief Row indices of training observations *not* in the bootstrap sample.
     *
     * Builds a set-based complement against `sample_indices` and returns
     * the sorted out-of-bag row indices in `[0, n_total)`.
     *
     * @param n_total  Total number of training observations.
     */
    std::vector<int> oob_indices(int n_total) const {
      std::set<int> const in_bag(sample_indices.begin(), sample_indices.end());
      std::vector<int> oob;
      oob.reserve(static_cast<std::size_t>(std::max(0, n_total - static_cast<int>(in_bag.size()))));
      for (int i = 0; i < n_total; ++i) {
        if (in_bag.find(i) == in_bag.end()) {
          oob.push_back(i);
        }
      }
      return oob;
    }

    /**
     * @brief Predict a subset of rows (typically OOB indices).
     *
     * The returned vector has the same size as @p row_idx; element `i`
     * is the wrapped model's prediction for row `row_idx[i]` of `x`.
     */
    types::OutcomeVector predict_oob(types::FeatureMatrix const& x, std::vector<int> const& row_idx) const {
      if (row_idx.empty()) {
        return types::OutcomeVector(0);
      }
      return model->predict(static_cast<types::FeatureMatrix>(x(row_idx, Eigen::all)));
    }

    /** @brief Whether the wrapped model reported a degenerate training run. */
    bool degenerate() const { return model->degenerate; }

    /**
     * @brief Structural equality on the wrapped model only.
     *
     * `sample_indices` is **deliberately excluded** from the comparison.
     * It records which rows were used to train this bag — bookkeeping for
     * OOB computation, not an identity property of the model. Two bags
     * that would produce the same predictions on every input are equal
     * here, even if they were trained on different bootstrap samples.
     *
     * Callers that need to assert sample-indices round-trip must compare
     * `sample_indices` directly (see `Json.test.cpp`'s round-trip test).
     */
    bool operator==(Bagged const& other) const { return *model == *other.model; }
    bool operator!=(Bagged const& other) const { return !(*this == other); }
  };
}
