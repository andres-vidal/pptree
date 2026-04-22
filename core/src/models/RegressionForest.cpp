#include "models/RegressionForest.hpp"

#include "models/Bagged.hpp"
#include "models/RegressionTree.hpp"
#include "models/VIVisitor.hpp"
#include "stats/RegressionMetrics.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
  using namespace ppforest2;
  using namespace ppforest2::types;
  using namespace ppforest2::stats;

  // Population variance of an OutcomeVector (n divisor so it matches the
  // scale of stats::mse: an "always-predict-mean" baseline gives NMSE = 1).
  double population_variance(OutcomeVector const& v) {
    int const n = static_cast<int>(v.size());
    if (n <= 0) {
      return 0.0;
    }

    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
      mean += static_cast<double>(v(i));
    }
    mean /= static_cast<double>(n);

    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
      double const d = static_cast<double>(v(i)) - mean;
      ss += d * d;
    }

    return ss / static_cast<double>(n);
  }

  /** @brief Uniform sample with replacement: n indices drawn from [0, n-1]. */
  std::vector<int> uniform_sample(int n, RNG& rng) {
    std::vector<int> indices;
    indices.reserve(n);

    Uniform const unif(0, n - 1);

    for (int j = 0; j < n; ++j) {
      indices.push_back(unif(rng));
    }

    std::sort(indices.begin(), indices.end());
    return indices;
  }

  /**
   * @brief Train one bootstrap-aggregated regression tree.
   *
   * Inlined here (not a separate class) so the whole forest-training
   * pipeline lives in one place. Previously this was
   * `RegressionBootstrapTree::train`; the subclass existed only to
   * namespace this routine. See the `Bagged<M>` template header for
   * the wrapper's role.
   */
  BaggedTree::Ptr train_regression_bag(
      TrainingSpec::Ptr const& training_spec, FeatureMatrix const& x, RNG& rng, OutcomeVector const& y
  ) {
    invariant(y.size() == x.rows(), "Response size must match x.rows() for regression");

    int const n_total = static_cast<int>(x.rows());

    std::vector<int> sample_indices = uniform_sample(n_total, rng);

    // Subsample x and y.
    FeatureMatrix sampled_x = x(sample_indices, Eigen::all);
    OutcomeVector sampled_y = y(sample_indices, Eigen::all).eval();

    // Sort sampled data by continuous response so ByCutpoint's
    // contiguous-block invariant holds at the root.
    int const n = static_cast<int>(sampled_y.size());
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&sampled_y](int a, int b) { return sampled_y(a) < sampled_y(b); });

    FeatureMatrix sorted_x(n, sampled_x.cols());
    OutcomeVector sorted_y(n);

    for (int i = 0; i < n; ++i) {
      sorted_x.row(i) = sampled_x.row(order[static_cast<std::size_t>(i)]);
      sorted_y(i)     = sampled_y(order[static_cast<std::size_t>(i)]);
    }

    // Build the initial median-split GroupPartition from the sorted response.
    GroupPartition sampled_gp = training_spec->init_groups(sorted_y);

    RegressionTree::Ptr tree = RegressionTree::train(*training_spec, sorted_x, sorted_y, sampled_gp, rng);

    return std::make_unique<BaggedTree>(std::move(tree), std::move(sample_indices));
  }
}

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  RegressionForest::RegressionForest() = default;

  RegressionForest::RegressionForest(TrainingSpec::Ptr training_spec)
      : Forest(std::move(training_spec)) {
    // Subclass-mode invariant: constructing a RegressionForest with a
    // classification spec silently routes predict / OOB / VI through the
    // wrong math. Fail at construction rather than downstream.
    invariant(
        !this->training_spec || this->training_spec->mode == types::Mode::Regression,
        "RegressionForest requires a TrainingSpec with mode = Regression"
    );
  }

  RegressionForest::Ptr
  RegressionForest::train(TrainingSpec const& training_spec, FeatureMatrix const& x, OutcomeVector const& y) {
    invariant(training_spec.mode == Mode::Regression, "RegressionForest::train requires mode = Regression");

    int const size        = training_spec.size;
    int const seed        = training_spec.seed;
    int const max_retries = training_spec.max_retries;

    user_error(size > 0, "Forest size must be greater than 0 (got " + std::to_string(size) + ").");

    // clang-format off
    #ifdef _OPENMP
    omp_set_num_threads(training_spec.resolve_threads());
    #endif
    // clang-format on

    // Note: we do not compute a forest-level `y_part` here —
    // `train_regression_bag` rebuilds the initial partition from the
    // sampled, resorted response (each tree's OOB subset has different
    // median-split boundaries from the full dataset).
    TrainingSpec::Ptr spec = TrainingSpec::make(training_spec);

    std::vector<BaggedTree::Ptr> boots(size);
    std::vector<std::exception_ptr> errors(size);

    // clang-format off
    #pragma omp parallel for schedule(static)
    // clang-format on
    for (int i = 0; i < size; ++i) {
      for (int attempt = 0; attempt <= max_retries; ++attempt) {
        try {
          // Stream id `i + attempt * size` partitions `[0, size*(max_retries+1))`
          // into disjoint blocks: attempt 0 uses ids [0, size), attempt 1 uses
          // [size, 2*size), etc. Each (tree_index, attempt) pair gets a unique
          // stream. This formula is load-bearing for golden-file reproducibility
          // — changing either the loop bounds or the stride invalidates every
          // regression golden file and breaks the determinism contract
          // documented in CLAUDE.md (reproducibility across platforms).
          uint64_t stream = static_cast<uint64_t>(i) + static_cast<uint64_t>(attempt) * static_cast<uint64_t>(size);
          RNG rng(static_cast<uint64_t>(seed), stream);
          boots[i] = train_regression_bag(spec, x, rng, y);

          if (!boots[i]->degenerate()) {
            break;
          }
        } catch (...) {
          errors[i] = std::current_exception();
          break;
        }
      }
    }

    auto forest = std::make_unique<RegressionForest>(spec);

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

  Outcome RegressionForest::predict(FeatureVector const& data) const {
    // Guard against an empty forest: `trees.size() == 0` would turn the
    // final division into 0 / 0 = NaN, which silently propagates as a
    // valid-looking prediction. Mirrors the classification counterpart's
    // `predict(FeatureMatrix, Proportions)` check.
    invariant(!trees.empty(), "Forest has no trees.");

    Feature sum = Feature(0);

    for (auto const& tree : trees) {
      sum += static_cast<Feature>(tree->predict(data));
    }

    return static_cast<Outcome>(sum / static_cast<Feature>(trees.size()));
  }

  FeatureMatrix RegressionForest::predict(FeatureMatrix const& /*data*/, Proportions) const {
    throw std::invalid_argument("Vote proportions are not available for regression forests. "
                                "Use predict(data) for numeric predictions.");
  }

  OutcomeVector RegressionForest::oob_predict(FeatureMatrix const& x) const {
    // Empty forest (zero trees) is a valid no-op for OOB prediction:
    // every row has count == 0, so every entry in the output is the
    // `no_oob` NaN sentinel. Downstream `oob_error` filters rows with
    // `std::isnan` and returns `-1.0` when no rows survive the filter,
    // preserving the "no OOB data" semantics without a hard failure.
    // This is deliberately asymmetric with `predict(FeatureVector)`,
    // which asserts `!trees.empty()` — prediction on zero trees is
    // undefined (there's no aggregation to perform), but "OOB stats over
    // zero trees" is well-defined (the empty average).
    int const n_total = static_cast<int>(x.rows());
    int const B       = static_cast<int>(trees.size());

    std::vector<Feature> sums(static_cast<std::size_t>(n_total), Feature(0));
    std::vector<int> counts(static_cast<std::size_t>(n_total), 0);

    for (int k = 0; k < B; ++k) {
      BaggedTree const& tree   = *trees[k];
      std::vector<int> oob_idx = tree.oob_indices(n_total);
      OutcomeVector preds      = tree.predict_oob(x, oob_idx);

      for (int j = 0; j < static_cast<int>(oob_idx.size()); ++j) {
        int i = oob_idx[static_cast<std::size_t>(j)];
        sums[static_cast<std::size_t>(i)] += static_cast<Feature>(preds(j));
        counts[static_cast<std::size_t>(i)] += 1;
      }
    }

    // Sentinel: NaN for observations with no OOB tree. -1 would collide with
    // valid regression predictions. Callers filter with std::isnan.
    Outcome const no_oob = std::numeric_limits<Outcome>::quiet_NaN();

    OutcomeVector out(n_total);

    for (int i = 0; i < n_total; ++i) {
      int cnt = counts[static_cast<std::size_t>(i)];
      out(i)  = cnt > 0 ? static_cast<Outcome>(sums[static_cast<std::size_t>(i)] / static_cast<Feature>(cnt)) : no_oob;
    }

    return out;
  }

  std::optional<double> RegressionForest::oob_error(FeatureMatrix const& x, OutcomeVector const& y) const {
    user_error(
        y.size() == x.rows(),
        "oob_error: response length (" + std::to_string(y.size()) +
            ") does not match the number of observations in x (" + std::to_string(x.rows()) + ")."
    );

    OutcomeVector preds = oob_predict(x);

    std::vector<int> oob_rows;
    for (int i = 0; i < preds.size(); ++i) {
      if (!std::isnan(static_cast<double>(preds(i)))) {
        oob_rows.push_back(i);
      }
    }

    if (oob_rows.empty()) {
      // See `ClassificationForest::oob_error` — nullopt signals "no OOB
      // data" cleanly at the C++/R/CLI boundaries.
      return std::nullopt;
    }

    double mse = 0.0;
    for (int i = 0; i < static_cast<int>(oob_rows.size()); ++i) {
      int const row     = oob_rows[static_cast<std::size_t>(i)];
      double const diff = static_cast<double>(preds(row)) - static_cast<double>(y(row));
      mse += diff * diff;
    }

    return mse / static_cast<double>(oob_rows.size());
  }

  void RegressionForest::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }

  // -------------------------------------------------------------------------
  // VI1 — NMSE-based permuted importance
  // -------------------------------------------------------------------------
  FeatureVector RegressionForest::vi_permuted(FeatureMatrix const& x, OutcomeVector const& y, int seed) const {
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
        // The tree saw every training row during fitting — no out-of-sample
        // view to evaluate against, so it can't participate in the OOB-based
        // VI average. Handled defensively: empty OOB is astronomically rare
        // under bootstrap sampling.
        continue;
      }

      int const n_oob = static_cast<int>(oob_idx.size());

      OutcomeVector y_oob(n_oob);
      for (int i = 0; i < n_oob; ++i) {
        y_oob(i) = y(oob_idx[static_cast<std::size_t>(i)]);
      }

      double const var_y = population_variance(y_oob);

      if (var_y <= 0.0) {
        // OOB rows all share the same `y`, so NMSE is undefined
        // (denominator 0). The tree has no usable out-of-sample signal for
        // ranking variables — drop it rather than invent a substitute.
        continue;
      }

      OutcomeVector baseline_pred = tree.predict_oob(x, oob_idx);
      double const baseline_mse   = stats::mse(baseline_pred, y_oob);

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

        double const perm_mse = stats::mse(perm_pred, y_oob);
        importance(j) += static_cast<Feature>((perm_mse - baseline_mse) / var_y);

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
  // VI3 — NMSE-weighted projections (no G-1 factor)
  // -------------------------------------------------------------------------
  FeatureVector RegressionForest::vi_weighted_projections(
      FeatureMatrix const& x, OutcomeVector const& y, FeatureVector const* scale
  ) const {
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
        // The tree saw every training row during fitting, so it has no
        // out-of-sample view and can't contribute an OOB quality weight.
        // Same reasoning as VI1 above.
        continue;
      }

      int const n_oob = static_cast<int>(oob_idx.size());
      OutcomeVector y_oob(n_oob);
      for (int i = 0; i < n_oob; ++i) {
        y_oob(i) = y(oob_idx[static_cast<std::size_t>(i)]);
      }

      double const var_y = population_variance(y_oob);

      if (var_y <= 0.0) {
        // OOB rows all share the same `y`, so NMSE is undefined. No
        // usable out-of-sample signal for a quality weight — drop the tree.
        continue;
      }

      OutcomeVector oob_preds = tree.predict_oob(x, oob_idx);
      double const nmse       = stats::mse(oob_preds, y_oob) / var_y;
      Feature const weight    = static_cast<Feature>(std::max(0.0, 1.0 - nmse));

      VIVisitor visitor(n_vars, scale);
      tree.model->root->accept(visitor);

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += weight * static_cast<Feature>(visitor.vi3_contributions[static_cast<std::size_t>(j)]);
      }

      valid_trees++;
    }

    if (valid_trees > 0) {
      importance /= static_cast<Feature>(valid_trees);
    }

    return importance;
  }
}
