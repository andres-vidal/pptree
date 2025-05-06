#pragma once

#include "Tree.hpp"

namespace models {
  template <typename T, typename R>
  class VIStrategy;

  template<typename T, typename R>
  struct BootstrapTree;

  template<typename T, typename R>
  using BootstrapTreePtr = std::unique_ptr<BootstrapTree<T, R> >;

  template<typename T, typename R>
  struct BootstrapTree : public Tree<T, R> {
    using Base = Tree<T, R>;
    using Base::Base;
    using Base::error_rate;
    using Base::confusion_matrix;

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
        x,
        y,
        tree.classes,
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
      TreeNodePtr<T, R>            root,
      TrainingSpecPtr<T, R>        training_spec,
      const stats::Data<T> &       x,
      const stats::DataColumn<R> & y,
      const std::set<R> &          classes,
      const std::vector<int> &     iob_indices,
      const std::set<int> &        oob_indices) :
      Base(
        std::move(root),
        std::move(training_spec),
        x,
        y,
        classes),
      iob_indices(iob_indices),
      oob_indices(oob_indices) {
    }

    BootstrapTreePtr<T, R> retrain(stats::Data<T> &x,  stats::DataColumn<R> &y, const std::vector<int> &iob_indices) const {
      return BootstrapTree<T, R>::train(*this->training_spec, x, y, iob_indices);
    }

    double error_rate() const {
      std::vector<int> oob_indices_vec(oob_indices.begin(), oob_indices.end());

      return error_rate(this->x(oob_indices_vec, Eigen::all), this->y(oob_indices_vec, Eigen::all));
    }

    stats::ConfusionMatrix confusion_matrix() const {
      std::vector<int> oob_indices_vec(oob_indices.begin(), oob_indices.end());

      return confusion_matrix(this->x(oob_indices_vec, Eigen::all), this->y(oob_indices_vec, Eigen::all));
    }

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }
  };
}
