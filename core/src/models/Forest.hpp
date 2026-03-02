#pragma once

#include "models/BootstrapTree.hpp"

#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace pptree {
  struct Forest {
    static Forest train(
      const TrainingSpec&    training_spec,
      types::FeatureMatrix&  x,
      types::ResponseVector& y,
      int                    size,
      int                    seed,
      int                    n_threads = std::thread::hardware_concurrency());

    std::vector<BootstrapTree::Ptr> trees;
    TrainingSpec::Ptr training_spec;
    const int seed = 0;

    Forest();
    Forest(TrainingSpec::Ptr&& training_spec, int seed);

    types::Response predict(const types::FeatureVector& data) const;
    types::ResponseVector predict(const types::FeatureMatrix& data) const;

    void add_tree(std::unique_ptr<BootstrapTree> tree);

    bool operator==(const Forest& other) const;
    bool operator!=(const Forest& other) const;

    private:
      types::Response predict(const types::FeatureVector& data, const std::vector<int>& indx) const;
  };
}
