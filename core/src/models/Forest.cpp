#include "models/Forest.hpp"
#include "models/ModelVisitor.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree::stats;
using namespace pptree::types;

namespace pptree {
  Forest Forest::train(
    const TrainingSpec &  training_spec,
    const FeatureMatrix & x,
    const ResponseVector& y,
    const int             size,
    const int             seed,
    const int             n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    GroupPartition group_spec(y);

    invariant(size > 0, "The forest size must be greater than 0.");

    std::vector<BootstrapTree::Ptr> trees(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      RNG rng(static_cast<uint64_t>(seed), static_cast<uint64_t>(i));
      trees[i] = BootstrapTree::train(training_spec, x, group_spec, rng);
    }

    Forest forest(training_spec.clone(), seed);

    for (auto& tree : trees) {
      forest.add_tree(std::move(tree));
    }

    return forest;
  }

  Forest::Forest() {
  }

  Forest::Forest(TrainingSpec::Ptr&& training_spec, int seed)
    : training_spec(std::move(training_spec)),
    seed(seed) {
  }

  Response Forest::predict(const FeatureVector& data) const {
    std::map<Response, int> votes_per_group;

    for (const auto& tree : trees) {
      Response prediction = tree->predict(data);
      votes_per_group[prediction] += 1;
    }

    Response best  = 0;
    int best_count = 0;

    for (const auto& [key, votes] : votes_per_group) {
      if (votes > best_count) {
        best       = key;
        best_count = votes;
      }
    }

    return best;
  }

  ResponseVector Forest::predict(const FeatureMatrix& data) const {
    ResponseVector predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((FeatureVector)data.row(i));
    }

    return predictions;
  }

  void Forest::add_tree(std::unique_ptr<BootstrapTree> tree) {
    trees.push_back(std::move(tree));
  }

  bool Forest::operator==(const Forest& other) const {
    if (trees.size() != other.trees.size()) {
      return false;
    }

    for (std::size_t i = 0; i < trees.size(); i++) {
      if (*trees[i] != *other.trees[i]) {
        return false;
      }
    }

    return true;
  }

  bool Forest::operator!=(const Forest& other) const {
    return !(*this == other);
  }

  void Forest::accept(ModelVisitor& visitor) const {
    visitor.visit(*this);
  }

  ResponseVector Forest::oob_predict(const FeatureMatrix& x) const {
    const int n_total = static_cast<int>(x.rows());
    const int B       = static_cast<int>(trees.size());

    std::vector<std::map<Response, int>> votes(static_cast<std::size_t>(n_total));

    for (int k = 0; k < B; ++k) {
      const BootstrapTree& tree    = *trees[k];
      std::vector<int>     oob_idx = tree.oob_indices(n_total);
      ResponseVector preds         = tree.predict_oob(x, oob_idx);

      for (int j = 0; j < static_cast<int>(oob_idx.size()); ++j) {
        int i = oob_idx[static_cast<std::size_t>(j)];
        votes[static_cast<std::size_t>(i)][preds(j)] += 1;
      }
    }

    ResponseVector out(n_total);
    out.fill(-1);

    for (int i = 0; i < n_total; ++i) {
      const auto& obs_votes = votes[static_cast<std::size_t>(i)];

      if (obs_votes.empty()) {
        continue;
      }

      Response best  = 0;
      int best_count = 0;

      for (const auto& [cls, cnt] : obs_votes) {
        if (cnt > best_count) {
          best       = cls;
          best_count = cnt;
        }
      }

      out(i) = best;
    }

    return out;
  }

  double Forest::oob_error(
    const FeatureMatrix&  x,
    const ResponseVector& y) const {
    ResponseVector preds = oob_predict(x);

    std::vector<int> oob_rows;

    for (int i = 0; i < preds.size(); ++i) {
      if (preds(i) >= 0) {
        oob_rows.push_back(i);
      }
    }

    if (oob_rows.empty()) {
      return -1.0;
    }

    ResponseVector preds_oob = preds(oob_rows, Eigen::all).eval();
    ResponseVector y_oob     = y(oob_rows, Eigen::all).eval();

    return stats::error_rate(preds_oob, y_oob);
  }
}
