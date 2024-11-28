#pragma once

#include "Node.hpp"
#include "SortedDataSpec.hpp"
#include "BootstrapDataSpec.hpp"
#include "TrainingSpec.hpp"
#include "ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>


namespace models {
  using json = nlohmann::json;

  template <typename T, typename R>
  class VIStrategy;

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  struct BaseTree {
    static DerivedTree<T, R> train(const TrainingSpec<T, R> &training_spec, const D &training_data);

    std::unique_ptr<Condition<T, R> > root;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;
    std::shared_ptr<D> training_data;

    explicit BaseTree(std::unique_ptr<Condition<T, R> > root) : root(std::move(root)) {
    }

    BaseTree(
      std::unique_ptr<Condition<T, R> >    root,
      std::unique_ptr<TrainingSpec<T, R> > training_spec,
      std::shared_ptr<D >                  training_data)
      : root(std::move(root)),
      training_spec(std::move(training_spec)),
      training_data(training_data) {
    }

    DerivedTree<T, R> retrain(const D &data) const {
      return BaseTree::train(*this->training_spec, data);
    }

    DerivedTree<T, R> standardize() const {
      return retrain(stats::center(descale(*training_data)));
    }

    R predict(const stats::DataColumn<T> &data) const {
      return root->predict(data);
    }

    stats::DataColumn<R> predict(const stats::Data<T> &data) const {
      stats::DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((stats::DataColumn<T>)data.row(i));
      }

      return predictions;
    }

    bool operator==(const BaseTree<T, R, D, DerivedTree> &other) const {
      return *root == *other.root;
    }

    bool operator!=(const BaseTree<T, R, D, DerivedTree> &other) const {
      return !(*this == other);
    }

    virtual double error_rate(const stats::SortedDataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::error_rate(predict(x), y);
    }

    virtual stats::ConfusionMatrix confusion_matrix(const stats::SortedDataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::ConfusionMatrix(predict(x), y);
    }

    json to_json() const {
      return json{
        { "root", root->to_json() }
      };
    }
  };

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  std::ostream& operator<<(std::ostream & ostream, const BaseTree<T, R, D, DerivedTree>& tree) {
    return ostream << tree.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  struct Tree : public BaseTree<T, R, stats::SortedDataSpec<T, R>, Tree> {
    using Base = BaseTree<T, R, stats::SortedDataSpec<T, R>, Tree >;
    using Base::Base;

    static Tree<T, R> train(const TrainingSpec<T, R> &training_spec, const stats::SortedDataSpec<T, R> &training_data) {
      return Base::train(training_spec, training_data);
    }

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }

    // cppcheck-suppress noExplicitConstructor
    Tree(Base tree)
      : Base(std::move(tree.root), std::move(tree.training_spec), std::move(tree.training_data)) {
    }
  };
}
