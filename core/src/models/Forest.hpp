#pragma once

#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "models/VariableImportance.hpp"

#include <memory>
#include <thread>
#include <vector>

namespace ppforest2 {
  /**
   * @brief Abstract base class for projection pursuit random forests.
   *
   * Holds a vector of `BaggedTree` wrappers (each pairs a `Tree` with the
   * bootstrap sample indices it was trained on) and the shared training
   * spec. Aggregation logic (majority vote vs mean), proportion
   * predictions, and OOB handling are defined in the concrete subclasses
   * `ClassificationForest` and `RegressionForest`.
   *
   * Construct via `Forest::train`, which dispatches to the correct
   * concrete type based on `training_spec.mode`:
   *
   * @code
   *   auto forest = Forest::train(spec, x, y);            // classification
   *   auto forest = Forest::train(spec, x, y, &cont_y);   // regression
   *   auto preds = forest->predict(x_test);
   *
   *   // oob_error is defined on the concrete subclass (it takes mode-specific
   *   // truth types — GroupIdVector for classification, OutcomeVector for
   *   // regression). Downcast to call it.
   *   auto& cf = dynamic_cast<ClassificationForest&>(*forest);
   *   double oob = cf.oob_error(x, y);
   * @endcode
   */
  struct Forest : public Model {
    using Ptr = std::unique_ptr<Forest>;

    /** @brief Bootstrap-aggregated trees. Each `BaggedTree` pairs the
     * polymorphic inner `Tree` with its sample indices. */
    std::vector<BaggedTree::Ptr> trees;

    /**
     * @brief Train a random forest.
     *
     * Dispatches to `ClassificationForest::train` or `RegressionForest::train`
     * based on `training_spec.mode`. Note that the top-level `x` / `y` are
     * not mutated here — each bootstrap tree resamples into its own local
     * storage. (Contrast with single-tree `Tree::train`, where regression
     * mode does permute `x` / `y` in place.)
     */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y
    );

    /** @brief Concrete — iterates rows and calls the virtual `predict(FeatureVector)`. */
    types::OutcomeVector predict(types::FeatureMatrix const& data) const override;

    /** @brief Per-row prediction (mode-specific: majority vote or mean). */
    types::Outcome predict(types::FeatureVector const& data) const override = 0;

    /**
     * @brief Per-group proportion predictions. Classification only.
     *
     * Regression subclasses throw.
     */
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override = 0;

    /**
     * @brief Out-of-bag predictions.
     *
     * For classification, returns majority-vote labels with `-1` as the
     * "no OOB tree" sentinel (labels are non-negative integers). For
     * regression, returns the mean of OOB tree predictions with `NaN`
     * as the sentinel (valid predictions can be any finite value).
     */
    virtual types::OutcomeVector oob_predict(types::FeatureMatrix const& x) const = 0;

    // Note: `oob_error` is NOT on the base class. Its truth-vector type differs
    // per mode (classification uses GroupIdVector, regression uses OutcomeVector),
    // so each concrete subclass defines its own `oob_error(x, truth)` with the
    // correct parameter type. Callers that hold a `Forest::Ptr` must downcast to
    // the concrete type (or, in R/CLI, dispatch on the spec's mode).

    /** @brief Accept a model visitor (mode-specific dispatch). */
    void accept(Model::Visitor& visitor) const override = 0;

    /**
     * @brief VI1 — per-variable permuted importance.
     *
     * Per tree: measures how much the OOB error-of-fit metric increases
     * when each predictor column is permuted among the OOB rows.
     * Classification: accuracy drop. Regression: NMSE increase (MSE
     * increase normalised by `Var(y_oob)`, making magnitudes comparable
     * across datasets). Averaged over non-degenerate trees.
     *
     * **Sign semantics.** Values may be **negative**. This is not a
     * bug and not a sentinel: it means that permuting the feature did
     * not degrade OOB fit on average — the feature's signal sits at or
     * below the noise floor of the permutation procedure. Callers
     * should interpret negative or near-zero entries as "no evidence
     * of importance" rather than clamping them to zero; the relative
     * ranking of values carries the intended information. Do not
     * apply additional normalization on the R / CLI side — the scale
     * is already comparable within a fitted model.
     */
    virtual types::FeatureVector vi_permuted(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        int seed
    ) const = 0;

    /**
     * @brief VI2 — projection-coefficient importance (averaged over trees).
     *
     * At every split node s with G_s groups, accumulates |a_j| / G_s.
     * When @p scale is provided each |a_j| is multiplied by σ_j.
     * Mode-agnostic — single implementation on the base.
     */
    types::FeatureVector
    vi_projections(int n_vars, types::FeatureVector const* scale = nullptr) const;

    /**
     * @brief VI3 — weighted projection-coefficient importance.
     *
     * Each tree's contribution is weighted by a per-tree OOB quality
     * score (classification: `1 - error_rate` in `[0, 1]`; regression:
     * `max(0, 1 - NMSE)` in `[0, 1]`), and each split contributes
     * `I_s × |a_j|` (non-negative). Entries are therefore always
     * non-negative by construction. Unlike `vi_permuted`, this measure
     * carries no noise-floor signal: a zero entry means "this feature
     * never appeared (weighted) in any OOB-contributing split," not
     * "within noise." Callers should rely on the ranking; do not
     * re-normalize at the R / CLI boundary.
     */
    virtual types::FeatureVector vi_weighted_projections(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        types::FeatureVector const* scale = nullptr
    ) const = 0;

    /**
     * @brief Convenience — compute all three VI measures at once.
     *
     * Derives `scale` from the columnwise standard deviation of @p x
     * (zeros replaced by 1 to guard against division-by-zero downstream),
     * then populates `permuted`, `projections`, and `weighted_projections`.
     */
    VariableImportance variable_importance(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        int seed
    ) const;

    /**
     * @brief Add a trained bagged tree to the forest.
     *
     * Asserts at runtime that the incoming tree's mode matches this
     * forest's mode (e.g., a `ClassificationForest` can only accept trees
     * trained under classification). This is the type-safety compromise
     * of keeping `Forest::trees` mode-agnostic at the container level:
     * the check runs at assembly time rather than at every prediction
     * site, but is cheaper than threading templates through every call
     * site that handles a `Forest::Ptr`.
     */
    void add_tree(BaggedTree::Ptr tree);

    bool operator==(Forest const& other) const;
    bool operator!=(Forest const& other) const;

  protected:
    Forest();
    explicit Forest(TrainingSpec::Ptr training_spec);
  };
}
