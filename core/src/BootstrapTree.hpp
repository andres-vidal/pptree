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
      const TrainingSpec<T, R> &         training_spec,
      const stats::SortedDataSpec<T, R> &training_data,
      const std::vector<int> &           iob_indices) {
      Tree<T, R> tree = Tree<T, R>::train(training_spec, training_data.select(iob_indices));

      std::set<int> oob_indices;
      std::set<int> iob_indices_set(iob_indices.begin(), iob_indices.end());

      for (int i = 0; i < training_data.x.rows(); i++) {
        if (iob_indices_set.count(i) == 0) {
          oob_indices.insert(i);
        }
      }

      return std::make_unique<BootstrapTree<T, R> >(
        std::move(tree.root),
        training_spec.clone(),
        training_data.x,
        training_data.y,
        training_data.classes,
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

    BootstrapTreePtr<T, R> retrain(const stats::SortedDataSpec<T, R> &data, const std::vector<int> &iob_indices) const {
      return BootstrapTree<T, R>::train(*this->training_spec, data, iob_indices);
    }

    stats::SortedDataSpec<T, R> get_oob() const {
      std::vector<int> oob_indices(this->oob_indices.begin(), this->oob_indices.end());

      return stats::SortedDataSpec<T, R>(
        this->x(oob_indices, Eigen::all),
        this->y(oob_indices, Eigen::all),
        this->classes);
    }

    float error_rate() const {
      return error_rate(this->get_oob());
    }

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }

    stats::ConfusionMatrix confusion_matrix() const {
      return confusion_matrix(this->get_oob());
    }
  };
}
