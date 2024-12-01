#pragma once

#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"

namespace models {
  template <typename T, typename R>
  class VIStrategy;

  template<typename T, typename R>
  struct BootstrapTree : public BaseTree<T, R, stats::BootstrapDataSpec<T, R>, BootstrapTree > {
    using Base = BaseTree<T, R, stats::BootstrapDataSpec<T, R>, BootstrapTree >;
    using Base::Base;
    using Base::error_rate;
    using Base::confusion_matrix;

    static BootstrapTree<T, R> train(const TrainingSpec<T, R> &training_spec, const stats::BootstrapDataSpec<T, R> &training_data) {
      return Base::train(training_spec, training_data);
    }

    explicit BootstrapTree(std::unique_ptr<Condition<T, R> > root)
      : Base(std::move(root)) {
    }

    // cppcheck-suppress noExplicitConstructor
    BootstrapTree(Base tree)
      : Base(std::move(tree.root), std::move(tree.training_spec), std::move(tree.training_data)) {
    }

    BootstrapTree(
      std::unique_ptr<Condition<T, R> >                root,
      std::unique_ptr<TrainingSpec<T, R> >             training_spec,
      std::shared_ptr<stats::BootstrapDataSpec<T, R> > training_data)
      : Base(std::move(root),  std::move(training_spec), training_data) {
    }

    double error_rate() const {
      return error_rate(this->training_data->get_oob());
    }

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }

    stats::ConfusionMatrix confusion_matrix() const {
      return confusion_matrix(this->training_data->get_oob());
    }
  };
}
