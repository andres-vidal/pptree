#pragma once

#include "Tree.hpp"

namespace models {
  template<typename T, typename R>
  struct BootstrapTree;

  template<typename T, typename R>
  using BootstrapTreePtr = std::unique_ptr<BootstrapTree<T, R> >;

  template<typename T, typename R>
  struct BootstrapTree : public Tree<T, R> {
    using Base = Tree<T, R>;
    using Base::Base;

    static BootstrapTreePtr<T, R> train(
      const TrainingSpec<T, R> & training_spec,
      const stats::Data<T> &     x,
      const stats::GroupSpec<R> &group_spec,
      stats::RNG &               rng) {
      std::vector<int> sample_indices;
      sample_indices.reserve(x.rows());

      for (const auto& group : group_spec.groups) {
        const int group_size = group_spec.group_size(group);
        const int min_index  = group_spec.group_start(group);
        const int max_index  = group_spec.group_end(group);

        const stats::Uniform unif(min_index, max_index);

        for (int j = 0; j < group_size; j++) {
          sample_indices.push_back(unif(rng));
        }
      }

      std::sort(sample_indices.begin(), sample_indices.end());

      stats::Data<T> sampled_x = x(sample_indices, Eigen::all);

      Tree<T, R> tree = Tree<T, R>::train(training_spec, sampled_x, group_spec, rng);

      return std::make_unique<BootstrapTree<T, R> >(
        std::move(tree.root),
        training_spec.clone(),
        sample_indices);
    }

    std::vector<int> sample_indices;

    BootstrapTree(
      TreeNodePtr<T, R> root) :
      Base(std::move(root)),
      sample_indices() {
    }

    BootstrapTree(
      TreeNodePtr<T, R>        root,
      TrainingSpecPtr<T, R>    training_spec,
      const std::vector<int> & sample_indices) :
      Base(std::move(root), std::move(training_spec)),
      sample_indices(sample_indices) {
    }
  };
}
