#pragma once

#include "Node.hpp"
#include "DataSpec.hpp"
#include "TrainingSpec.hpp"
#include "Projector.hpp"
#include "ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>

namespace models {
  using json = nlohmann::json;
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

    math::DVector<T> variable_importance() const {
      return variable_importance(VariableImportanceKind::PROJECTOR);
    }

    virtual long double error_rate(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::error_rate(predict(x), y);
    }

    virtual stats::ConfusionMatrix confusion_matrix(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::ConfusionMatrix(predict(x), y);
    }

    json to_json() const {
      return json{
        { "root", root->to_json() }
      };
    }

    protected:
      virtual math::DVector<T> variable_importance(VariableImportanceKind importance_kind) const {
        if (importance_kind == VariableImportanceKind::PERMUTATION) {
          throw std::invalid_argument("VariableImportanceKind::PERMUTATION not supported for a single BaseTree");
        }

        DerivedTree<T, R> std_tree = retrain(center(descale(*training_data)));

        long double factor = 1.0;

        if (importance_kind == VariableImportanceKind::PROJECTOR_ADJUSTED) {
          factor = 1.0 / (long double)(root->partition_count());
        }

        return std_tree.root->variable_importance(importance_kind) * factor;
      }
  };

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  std::ostream& operator<<(std::ostream & ostream, const BaseTree<T, R, D, DerivedTree>& tree) {
    return ostream << tree.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  struct Tree : public BaseTree<T, R, stats::DataSpec<T, R>, Tree> {
    using Base = BaseTree<T, R, stats::DataSpec<T, R>, Tree >;
    using Base::Base;

    static Tree<T, R> train(const TrainingSpec<T, R> &training_spec, const stats::DataSpec<T, R> &training_data) {
      return Base::train(training_spec, training_data);
    }

    // cppcheck-suppress noExplicitConstructor
    Tree(Base tree)
      : Base(std::move(tree.root), std::move(tree.training_spec), std::move(tree.training_data)) {
    }
  };
}
