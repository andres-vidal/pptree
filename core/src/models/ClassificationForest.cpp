#include "models/ClassificationForest.hpp"

#include "models/Bagged.hpp"
#include "models/ClassificationTree.hpp"
#include "models/VIVisitor.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  namespace {
    /**
     * @brief Stratified per-group sample: draw `group_size(g)` indices
     *        uniformly from each group `g` in the training data.
     *
     * Preserves the class balance across bootstrap replicates.
     */
    std::vector<int> stratified_sample(FeatureMatrix const& /*x*/, GroupPartition const& y_part, RNG& rng) {
      std::vector<int> indices;

      for (auto const& group : y_part.groups) {
        int const group_size = y_part.group_size(group);
        int const min_index  = y_part.group_start(group);
        int const max_index  = y_part.group_end(group);

        Uniform const unif(min_index, max_index);

        for (int j = 0; j < group_size; ++j) {
          indices.push_back(unif(rng));
        }
      }

      std::sort(indices.begin(), indices.end());
      return indices;
    }

    /**
     * @brief Train one bootstrap-aggregated classification tree.
     *
     * Inlined here (not a separate class) so the whole forest-training
     * pipeline lives in one place. Previously this was
     * `ClassificationBootstrapTree::train`, but that subclass existed
     * only to namespace this routine and duplicated a mode tag already
     * carried on the inner `Tree`. See the `Bagged<M>` template header
     * for the wrapper's role.
     */
    BaggedTree::Ptr train_classification_bag(
        TrainingSpec::Ptr const& training_spec,
        FeatureMatrix const& x,
        OutcomeVector const& y,
        GroupPartition const& y_part,
        RNG& rng
    ) {
      // Reusing the parent's `y_part` over the resampled `sampled_x`
      // only works if the spec's first block starts at row 0 and the
      // blocks cover `[0, x.rows())` contiguously. Both hold for
      // `init_groups` / `ByLabel::init` output today; guard explicitly so
      // any future change fails loudly.
      if (!y_part.groups.empty()) {
        GroupId const first = *y_part.groups.begin();
        invariant(y_part.group_start(first) == 0, "classification bag: y_part must start at row 0");
      }

      std::vector<int> sample_indices = stratified_sample(x, y_part, rng);
      FeatureMatrix sampled_x         = x(sample_indices, Eigen::all);
      OutcomeVector sampled_y         = y(sample_indices);

      ClassificationTree::Ptr tree = ClassificationTree::train(*training_spec, sampled_x, sampled_y, y_part, rng);

      return std::make_unique<BaggedTree>(std::move(tree), std::move(sample_indices));
    }
  }

  ClassificationForest::ClassificationForest() = default;

  ClassificationForest::ClassificationForest(TrainingSpec::Ptr training_spec)
      : Forest(std::move(training_spec)) {
    // See the mirror check in RegressionForest: mode/class mismatches
    // silently corrupt downstream math, so fail loudly at construction.
    invariant(
        !this->training_spec || this->training_spec->mode == types::Mode::Classification,
        "ClassificationForest requires a TrainingSpec with mode = Classification"
    );
  }

  ClassificationForest::Ptr
  ClassificationForest::train(TrainingSpec const& training_spec, FeatureMatrix const& x, OutcomeVector const& y) {
    invariant(training_spec.mode == Mode::Classification, "ClassificationForest::train requires mode = Classification");

    int const size        = training_spec.size;
    int const seed        = training_spec.seed;
    int const max_retries = training_spec.max_retries;

    user_error(size > 0, "Forest size must be greater than 0 (got " + std::to_string(size) + ").");

    // clang-format off
    #ifdef _OPENMP
    omp_set_num_threads(training_spec.resolve_threads());
    #endif
    // clang-format on

    GroupPartition y_part  = training_spec.init_groups(y);
    TrainingSpec::Ptr spec = TrainingSpec::make(training_spec);

    std::vector<BaggedTree::Ptr> boots(size);
    std::vector<std::exception_ptr> errors(size);

    // clang-format off
    #pragma omp parallel for schedule(static)
    // clang-format on
    for (int i = 0; i < size; ++i) {
      for (int attempt = 0; attempt <= max_retries; ++attempt) {
        try {
          // See the matching comment in RegressionForest::train: this stream
          // formula is load-bearing for golden-file reproducibility; don't
          // change the stride without regenerating every classification
          // golden and updating both forests in lockstep.
          uint64_t stream = static_cast<uint64_t>(i) + static_cast<uint64_t>(attempt) * static_cast<uint64_t>(size);
          RNG rng(static_cast<uint64_t>(seed), stream);
          boots[i] = train_classification_bag(spec, x, y, y_part, rng);

          if (!boots[i]->degenerate()) {
            break;
          }
        } catch (...) {
          errors[i] = std::current_exception();
          break;
        }
      }
    }

    auto forest = std::make_unique<ClassificationForest>(spec);

    for (int i = 0; i < size; ++i) {
      if (errors[i]) {
        std::rethrow_exception(errors[i]);
      }

      if (boots[i]->degenerate()) {
        forest->degenerate = true;
      }

      forest->add_tree(std::move(boots[i]));
    }

    return forest;
  }

  Outcome ClassificationForest::predict(FeatureVector const& data) const {
    // Match RegressionForest::predict(FeatureVector) — an empty forest has
    // no basis for prediction, and the pre-guard code silently returned
    // `Outcome(0)` regardless of the input.
    invariant(!trees.empty(), "Forest has no trees.");

    std::map<GroupId, int> votes_per_group;

    for (auto const& tree : trees) {
      GroupId prediction = static_cast<GroupId>(tree->predict(data));
      votes_per_group[prediction] += 1;
    }

    GroupId best   = 0;
    int best_count = 0;

    for (auto const& [key, votes] : votes_per_group) {
      if (votes > best_count) {
        best       = key;
        best_count = votes;
      }
    }

    return static_cast<Outcome>(best);
  }

  FeatureMatrix ClassificationForest::predict(FeatureMatrix const& data, Proportions) const {
    invariant(!trees.empty(), "Forest has no trees.");

    std::set<GroupId> group_set = trees[0]->model->root->node_groups();
    std::vector<GroupId> groups(group_set.begin(), group_set.end());
    int const G = static_cast<int>(groups.size());

    std::map<GroupId, int> group_to_col;
    for (int g = 0; g < G; ++g) {
      group_to_col[groups[static_cast<std::size_t>(g)]] = g;
    }

    int const n               = static_cast<int>(data.rows());
    FeatureMatrix proportions = FeatureMatrix::Zero(n, G);

    for (int i = 0; i < n; ++i) {
      for (auto const& tree : trees) {
        GroupId pred = static_cast<GroupId>(tree->predict(static_cast<FeatureVector>(data.row(i))));
        auto it      = group_to_col.find(pred);

        if (it != group_to_col.end()) {
          proportions(i, it->second) += 1;
        }
      }

      Feature total = proportions.row(i).sum();

      if (total > 0) {
        proportions.row(i) /= total;
      }
    }

    return proportions;
  }

  OutcomeVector ClassificationForest::oob_predict(FeatureMatrix const& x) const {
    // Empty forest is a valid no-op for OOB prediction: every row has
    // zero OOB trees voting, so every entry gets the `-1` sentinel.
    // Downstream `oob_error` filters out sentinels and returns
    // `std::nullopt` when no rows survive, preserving the "no OOB data"
    // semantics without a hard failure. Deliberately asymmetric with
    // `predict(FeatureVector)`, which asserts `!trees.empty()` —
    // aggregating zero votes into a single prediction has no defined
    // answer, but "OOB stats over zero trees" is well-defined (the
    // empty average). Mirrors `RegressionForest::oob_predict`.
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(trees.size());

    // Majority vote of OOB predictions per observation.
    std::vector<std::map<GroupId, int>> votes(static_cast<std::size_t>(n_total));

    for (int k = 0; k < B; ++k) {
      BaggedTree const& tree   = *trees[k];
      std::vector<int> oob_idx = tree.oob_indices(n_total);
      OutcomeVector preds      = tree.predict_oob(x, oob_idx);

      for (int j = 0; j < static_cast<int>(oob_idx.size()); ++j) {
        int i = oob_idx[static_cast<std::size_t>(j)];
        votes[static_cast<std::size_t>(i)][static_cast<GroupId>(preds(j))] += 1;
      }
    }

    // Sentinel -1 for observations with no OOB tree.
    OutcomeVector out(n_total);
    out.fill(-1);

    for (int i = 0; i < n_total; ++i) {
      auto const& obs_votes = votes[static_cast<std::size_t>(i)];

      if (obs_votes.empty()) {
        continue;
      }

      GroupId best   = 0;
      int best_count = 0;

      for (auto const& [cls, cnt] : obs_votes) {
        if (cnt > best_count) {
          best       = cls;
          best_count = cnt;
        }
      }

      out(i) = static_cast<Outcome>(best);
    }

    return out;
  }

  std::optional<double> ClassificationForest::oob_error(FeatureMatrix const& x, OutcomeVector const& y) const {
    OutcomeVector preds = oob_predict(x);

    std::vector<int> oob_rows;
    for (int i = 0; i < preds.size(); ++i) {
      if (preds(i) != Outcome(-1)) {
        oob_rows.push_back(i);
      }
    }

    if (oob_rows.empty()) {
      // No observation had any OOB tree — the OOB error is undefined.
      // Returning `std::nullopt` (instead of a -1.0 sentinel) lets the
      // caller distinguish "no data" from "real value", and flows
      // cleanly through to R as `NA_real_` / CLI as "not available".
      return std::nullopt;
    }

    OutcomeVector preds_oob = preds(oob_rows, Eigen::all).eval();
    // Subset first, cast second — avoids materializing a full-size int
    // vector just to read the OOB slice.
    GroupIdVector y_oob = y(oob_rows, Eigen::all).cast<GroupId>().eval();

    return stats::error_rate(preds_oob, y_oob);
  }

  void ClassificationForest::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }

  // -------------------------------------------------------------------------
  // VI1 — permuted importance (accuracy drop on OOB rows)
  // -------------------------------------------------------------------------
  FeatureVector ClassificationForest::vi_permuted(FeatureMatrix const& x, OutcomeVector const& y, int seed) const {
    GroupIdVector const y_int = y.cast<GroupId>();

    int const n_vars  = static_cast<int>(x.cols());
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(trees.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (int k = 0; k < B; ++k) {
      BaggedTree const& tree = *trees[k];

      if (tree.degenerate()) {
        continue;
      }

      std::vector<int> oob_idx = tree.oob_indices(n_total);

      if (oob_idx.empty()) {
        // The tree saw every training row during fitting — it has no
        // out-of-sample view to evaluate against. Including it in the
        // OOB-based VI average would mean mixing in-sample evidence with
        // out-of-sample evidence, same as counting a cross-validation
        // fold with no test data. Skip it entirely. Handled defensively:
        // empty OOB is astronomically rare under bootstrap sampling.
        continue;
      }

      int const n_oob = static_cast<int>(oob_idx.size());

      GroupIdVector oob_labels(n_oob);
      for (int i = 0; i < n_oob; ++i) {
        oob_labels(i) = y_int(oob_idx[static_cast<std::size_t>(i)]);
      }

      OutcomeVector baseline_pred = tree.predict_oob(x, oob_idx);
      float const baseline_acc    = stats::accuracy(baseline_pred, oob_labels);

      stats::RNG rng(static_cast<unsigned>(seed) ^ static_cast<unsigned>(k));
      stats::Uniform uniform(0, n_oob - 1);

      FeatureMatrix perm_x(n_oob, n_vars);
      for (int i = 0; i < n_oob; ++i) {
        perm_x.row(i) = x.row(oob_idx[static_cast<std::size_t>(i)]);
      }

      OutcomeVector perm_pred(n_oob);

      for (int j = 0; j < n_vars; ++j) {
        FeatureVector col_saved    = perm_x.col(j);
        std::vector<int> row_order = uniform.distinct(n_oob, rng);

        for (int i = 0; i < n_oob; ++i) {
          perm_x(i, j) = col_saved(row_order[static_cast<std::size_t>(i)]);
        }

        for (int i = 0; i < n_oob; ++i) {
          perm_pred(i) = tree.predict(static_cast<FeatureVector>(perm_x.row(i)));
        }

        float const perm_acc = stats::accuracy(perm_pred, oob_labels);
        importance(j) += baseline_acc - perm_acc;

        perm_x.col(j) = col_saved;
      }

      valid_trees++;
    }

    if (valid_trees > 0) {
      importance /= static_cast<Feature>(valid_trees);
    }

    return importance;
  }

  // -------------------------------------------------------------------------
  // VI3 — weighted projections (weight = 1 - OOB error rate)
  // -------------------------------------------------------------------------
  FeatureVector ClassificationForest::vi_weighted_projections(
      FeatureMatrix const& x, OutcomeVector const& y, FeatureVector const* scale
  ) const {
    GroupIdVector const y_int = y.cast<GroupId>();

    int const n_vars  = static_cast<int>(x.cols());
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(trees.size());

    std::set<GroupId> group_set;
    for (int i = 0; i < y_int.size(); ++i) {
      group_set.insert(y_int(i));
    }
    int const G = static_cast<int>(group_set.size());

    FeatureVector importance = FeatureVector::Zero(n_vars);
    int valid_trees          = 0;

    for (int k = 0; k < B; ++k) {
      BaggedTree const& tree = *trees[k];

      if (tree.degenerate()) {
        continue;
      }

      std::vector<int> oob_idx = tree.oob_indices(n_total);

      if (oob_idx.empty()) {
        // The tree saw every training row during fitting, so it has no
        // out-of-sample view and can't participate in the OOB-weighted
        // average. Same reasoning as VI1 above.
        continue;
      }

      int const n_oob = static_cast<int>(oob_idx.size());
      GroupIdVector oob_labels(n_oob);
      for (int i = 0; i < n_oob; ++i) {
        oob_labels(i) = y_int(oob_idx[static_cast<std::size_t>(i)]);
      }
      OutcomeVector oob_preds = tree.predict_oob(x, oob_idx);
      Feature const e_k       = static_cast<Feature>(stats::error_rate(oob_preds, oob_labels));

      VIVisitor visitor(n_vars, scale);
      tree.model->root->accept(visitor);

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
