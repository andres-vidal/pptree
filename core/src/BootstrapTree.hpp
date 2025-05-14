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
      const TrainingSpec<T, R> &  training_spec,
      const stats::Data<T> &      x,
      const stats::DataColumn<R> &y,
      const std::vector<int> &    iob_indices) {
      stats::Data<T> sampled_x       = x(iob_indices, Eigen::all);
      stats::DataColumn<R> sampled_y = y(iob_indices, Eigen::all);

      Tree<T, R> tree = Tree<T, R>::train(training_spec, sampled_x, sampled_y);

      std::set<int> oob_indices;
      std::set<int> iob_indices_set(iob_indices.begin(), iob_indices.end());

      for (int i = 0; i < x.rows(); i++) {
        if (iob_indices_set.count(i) == 0) {
          oob_indices.insert(i);
        }
      }

      return std::make_unique<BootstrapTree<T, R> >(
        std::move(tree.root),
        training_spec.clone(),
        iob_indices,
        oob_indices);
    }

    std::vector<int> iob_indices;
    std::set<int> oob_indices;

    BootstrapTree(
      TreeNodePtr<T, R> root) :
      Base(std::move(root)),
      iob_indices(),
      oob_indices() {
    }

    BootstrapTree(
      TreeNodePtr<T, R>        root,
      TrainingSpecPtr<T, R>    training_spec,
      const std::vector<int> & iob_indices,
      const std::set<int> &    oob_indices) :
      Base(std::move(root), std::move(training_spec)),
      iob_indices(iob_indices),
      oob_indices(oob_indices) {
    }
  };
}
