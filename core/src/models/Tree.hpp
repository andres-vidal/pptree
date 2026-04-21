#pragma once

#include "utils/Types.hpp"

#include "models/Bagged.hpp"
#include "models/Model.hpp"
#include "models/TreeNode.hpp"
#include "models/VariableImportance.hpp"


namespace ppforest2 {
  /**
   * @brief Abstract base class for projection pursuit decision trees.
   *
   * Each internal node projects data onto a linear combination of
   * features and splits on the projected value. Leaf values depend on
   * the mode — group labels for classification, mean response for
   * regression — implemented in the concrete subclasses
   * `ClassificationTree` and `RegressionTree`.
   *
   * Construct via the static `Tree::train` factory, which dispatches
   * to the correct concrete type based on `training_spec.mode`.
   *
   * @code
   *   auto tree = Tree::train(spec, x, y);  // classification or regression
   *   Outcome label = tree->predict(x.row(0));
   *   OutcomeVector preds = tree->predict(x);
   * @endcode
   */
  struct Tree : public Model {
    using Ptr = std::unique_ptr<Tree>;

    /** @brief Root node of the tree. */
    TreeNode::Ptr root;

    /**
     * @brief Train a tree from a response vector.
     *
     * Dispatches to `ClassificationTree::train` or `RegressionTree::train`
     * based on `training_spec.mode`. Creates an RNG from the spec's seed.
     *
     * `x` and `y` are taken by mutable reference because some strategies —
     * notably `ByCutpoint` for regression — permute rows in place during
     * training. Classification training does not mutate them, so the
     * classification path pays no cost for this signature. The alternative
     * (const-correct public API + defensive copy at the regression dispatch)
     * would force a full-matrix copy per single-tree regression call, which
     * is a real cost the library shouldn't absorb when the natural callers
     * (R bindings, CLI) discard the data right after training anyway.
     * Callers who need to preserve the original row order must copy before
     * calling.
     *
     * @param training_spec  Training specification (strategies + mode).
     * @param x              Feature matrix (n × p). May be permuted during regression training.
     * @param y              Response vector (n) — integer labels for
     *                       classification, continuous response for regression.
     *                       May be permuted during regression training.
     * @return               Trained tree as a polymorphic pointer.
     */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        types::OutcomeVector& y
    );

    /** @copydoc Tree::train(TrainingSpec const&, FeatureMatrix&, OutcomeVector&) */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        types::OutcomeVector& y,
        stats::RNG& rng
    );

    /** @copydoc Tree::train(TrainingSpec const&, FeatureMatrix&, OutcomeVector&) */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        stats::GroupPartition const& group_spec,
        types::OutcomeVector* y_vec = nullptr
    );

    /** @copydoc Tree::train(TrainingSpec const&, FeatureMatrix&, OutcomeVector&) */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        stats::GroupPartition const& group_spec,
        stats::RNG& rng,
        types::OutcomeVector* y_vec = nullptr
    );

    /**
     * @brief Predict a single observation.
     *
     * Walks the tree and returns the leaf value. Same implementation
     * for both modes — the leaf value is produced by the mode-specific
     * leaf strategy during training.
     */
    types::Outcome predict(types::FeatureVector const& data) const override;

    /** @brief Predict each row of a feature matrix. */
    types::OutcomeVector predict(types::FeatureMatrix const& data) const override;

    /**
     * @brief Predict per-group proportions. Mode-specific.
     *
     * Classification: one-hot encoding of the predicted group.
     * Regression: not available — throws.
     */
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override = 0;

    /** @brief Accept a model visitor (mode-specific dispatch). */
    void accept(Model::Visitor& visitor) const override = 0;

    /**
     * @brief VI2 — projection-coefficient importance for this tree.
     *
     * Accumulates |a_j| / G_s at every split node. When @p scale is
     * provided, |a_j| is multiplied by σ_j so coefficients are
     * comparable across variables on different scales. Mode-agnostic.
     */
    types::FeatureVector
    vi_projections(int n_vars, types::FeatureVector const* scale = nullptr) const;

    /**
     * @brief Convenience — bundle the available VI measures for a single tree.
     *
     * Populates `projections` and `scale`; `permuted` and
     * `weighted_projections` are left empty (forest-only measures).
     * `scale` is derived from the columnwise standard deviation of @p x.
     */
    VariableImportance variable_importance(types::FeatureMatrix const& x) const;

    bool operator==(Tree const& other) const;
    bool operator!=(Tree const& other) const;

  protected:
    Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    /**
     * @brief Build the root node of a tree.
     *
     * Iteratively grows the tree from the given group partition. Shared
     * implementation used by `ClassificationTree::train` and
     * `RegressionTree::train`.
     *
     * For regression, callers pass `mutable_x` and `mutable_y` pointing at
     * mutable copies they own, so the `ByCutpoint` grouping strategy can
     * reorder rows in place. For classification, both are null.
     *
     * @param spec        Training specification (strategies + mode).
     * @param x           Feature matrix (immutable view for classification,
     *                    or the same buffer as `*mutable_x` for regression).
     * @param y           Initial group partition for the root node.
     * @param rng         Random number generator (tree-local).
     * @param mutable_x   Optional mutable pointer to `x` for in-place reorder.
     * @param mutable_y   Optional mutable pointer to the response vector.
     * @return            Root `TreeNode` of the constructed tree.
     */
    static TreeNode::Ptr build_root(
        TrainingSpec const& spec,
        types::FeatureMatrix const& x,
        stats::GroupPartition const& y,
        stats::RNG& rng,
        types::FeatureMatrix* mutable_x = nullptr,
        types::OutcomeVector* mutable_y = nullptr
    );
  };

  /**
   * @brief Alias for the dominant `Bagged` instantiation in this codebase —
   * a bootstrap-aggregated `Tree`. Inner tree is polymorphic (classification
   * or regression via the `Tree` base); the wrapper itself is mode-agnostic.
   */
  using BaggedTree = Bagged<Tree>;
}
