#include "models/Forest.hpp"

#include "models/ClassificationForest.hpp"
#include "models/RegressionForest.hpp"
#include "models/VIVisitor.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"

using namespace ppforest2::types;

namespace ppforest2 {
  FeatureVector Forest::vi_projections(int n_vars, FeatureVector const* scale) const {
    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (auto const& bt : trees) {
      if (bt->degenerate()) {
        continue;
      }

      VIVisitor visitor(n_vars, scale);
      bt->model->root->accept(visitor);

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

  VariableImportance Forest::variable_importance(
      FeatureMatrix const& x, OutcomeVector const& y, int seed
  ) const {
    VariableImportance vi;
    vi.scale                = stats::sd(x);
    vi.scale                = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));
    vi.permuted             = vi_permuted(x, y, seed);
    vi.projections          = vi_projections(static_cast<int>(x.cols()), &vi.scale);
    vi.weighted_projections = vi_weighted_projections(x, y, &vi.scale);
    return vi;
  }


  Forest::Forest() = default;

  Forest::Forest(TrainingSpec::Ptr training_spec) {
    this->training_spec = std::move(training_spec);
  }

  OutcomeVector Forest::predict(FeatureMatrix const& data) const {
    OutcomeVector predictions(data.rows());

    for (int i = 0; i < data.rows(); ++i) {
      predictions(i) = predict(static_cast<FeatureVector>(data.row(i)));
    }

    return predictions;
  }

  void Forest::add_tree(BaggedTree::Ptr tree) {
    // Runtime mode-consistency check. `Forest::trees` is mode-erased at
    // the container level (to keep the R/JSON/CLI API surface non-
    // templated), so the compiler won't stop a `RegressionTree` from
    // being pushed into a `ClassificationForest` via the wrong caller.
    //
    // Scope of this check: it guards against caller-side mistakes that
    // mix trees from different specs — e.g. a test that accidentally
    // `add_tree`s a regression bag into a classification forest, or a
    // future training path that forks the spec partway through. It
    // does *not* provide independent validation against the JSON shape:
    // the deserialization path shares one `TrainingSpec::Ptr` between
    // the forest and all its inner trees, so on that path the check
    // passes by construction. Structural JSON validation lives in
    // `validate_forest_export` / `validate_tree_export`, which run
    // before this point.
    //
    // `this->training_spec` may be null on a default-constructed Forest
    // (used by the bare-model JSON deserialization path before a spec
    // is attached); in that case there's nothing to compare against.
    if (this->training_spec && tree && tree->model && tree->model->training_spec) {
      invariant(
          tree->model->training_spec->mode == this->training_spec->mode,
          "Forest::add_tree: tree mode does not match forest mode "
          "(attempted to add a tree trained under a different mode)"
      );
    }
    trees.push_back(std::move(tree));
  }

  bool Forest::operator==(Forest const& other) const {
    if (trees.size() != other.trees.size()) {
      return false;
    }

    for (std::size_t i = 0; i < trees.size(); ++i) {
      if (*trees[i] != *other.trees[i]) {
        return false;
      }
    }

    return true;
  }

  bool Forest::operator!=(Forest const& other) const {
    return !(*this == other);
  }

  Forest::Ptr Forest::train(
      TrainingSpec const& training_spec,
      FeatureMatrix const& x,
      OutcomeVector const& y
  ) {
    if (training_spec.mode == types::Mode::Regression) {
      return RegressionForest::train(training_spec, x, y);
    }

    return ClassificationForest::train(training_spec, x, y);
  }

}
