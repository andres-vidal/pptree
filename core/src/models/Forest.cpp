#include "models/Forest.hpp"
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
    const TrainingSpec & training_spec,
    FeatureMatrix &      x,
    ResponseVector &     y,
    const int            size,
    const int            seed,
    const int            n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    if (!GroupPartition::is_contiguous(y)) {
      sort(x, y);
    }

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
    std::vector<int> indx(trees.size());
    std::iota(indx.begin(), indx.end(), 0);
    return predict(data, indx);
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

  Response Forest::predict(
    const FeatureVector&    data,
    const std::vector<int>& indx) const {
    std::map<Response, int> votes_per_group;

    for (const auto& i : indx) {
      Response prediction = trees[i]->predict(data);
      votes_per_group[prediction] += 1;
    }

    int most_voted_group_votes = 0;
    Response most_voted_group  = 0;

    for (const auto& [key, votes] : votes_per_group) {
      if (votes > most_voted_group_votes) {
        most_voted_group       = key;
        most_voted_group_votes = votes;
      }
    }

    return most_voted_group;
  }
}
