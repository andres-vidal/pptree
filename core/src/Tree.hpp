#pragma once

#include "TreeNode.hpp"
#include "TreeCondition.hpp"
#include "TreeResponse.hpp"

#include "TrainingSpec.hpp"
#include "ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>


namespace models {
  using json = nlohmann::json;

  template <typename T, typename R>
  class VIStrategy;

  template<typename T, typename R>
  struct Tree {
    static Tree<T, R> train(
      const TrainingSpec<T, R> &  training_spec,
      const stats::Data<T>&       x,
      const stats::DataColumn<R>& y);

    TreeNodePtr<T, R> root;
    TrainingSpecPtr<T, R> training_spec;

    const stats::Data<T> x;
    const stats::DataColumn<R> y;
    const std::set<R> classes;

    Tree(
      TreeNodePtr<T, R> root) :
      root(std::move(root)),
      training_spec(TrainingSpecGLDA<T, R>::make(0.5)),
      x(stats::Data<T>()),
      y(stats::DataColumn<R>()),
      classes(std::set<R>()) {
    }

    Tree(
      TreeNodePtr<T, R>            root,
      TrainingSpecPtr<T, R>        training_spec,
      const stats::Data<T> &       x,
      const stats::DataColumn<R> & y,
      const std::set<R> &          classes) :

      root(std::move(root)),
      training_spec(std::move(training_spec)),
      x(x),
      y(y),
      classes(classes) {
    }

    Tree<T, R> retrain(const stats::Data<T> &x, const stats::DataColumn<R> &y) const {
      return Tree<T, R>::train(*this->training_spec, x, y);
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

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }

    bool operator==(const Tree<T, R> &other) const {
      return *root == *other.root;
    }

    bool operator!=(const Tree<T, R> &other) const {
      return !(*this == other);
    }

    virtual float error_rate(const stats::Data<T> &x, const stats::DataColumn<R> &y) const {
      return stats::error_rate(predict(x), y);
    }

    virtual stats::ConfusionMatrix confusion_matrix(const stats::Data<T> &x, const stats::DataColumn<R> &y) const {
      return stats::ConfusionMatrix(predict(x), y);
    }

    json to_json() const {
      return json{
        { "root", root->to_json() }
      };
    }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Tree<T, R>& tree) {
    return ostream << tree.to_json().dump(2, ' ', false);
  }
}
