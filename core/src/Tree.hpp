#pragma once

#include "TreeNode.hpp"
#include "TreeCondition.hpp"
#include "TreeResponse.hpp"

#include "TrainingSpec.hpp"
#include "ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>


namespace models {
  using json = nlohmann::json;

  template<typename T, typename R>
  struct Tree {
    static Tree<T, R> train(
      const TrainingSpec<T, R> & training_spec,
      stats::Data<T>&            x,
      stats::DataColumn<R>&      y,
      stats::RNG &               rng);

    static Tree<T, R> train(
      const TrainingSpec<T, R> & training_spec,
      stats::Data<T>&            x,
      const stats::GroupSpec<R>& group_spec,
      stats::RNG &               rng);

    TreeNodePtr<T, R> root;
    TrainingSpecPtr<T, R> training_spec;

    Tree(
      TreeNodePtr<T, R> root) :
      root(std::move(root)),
      training_spec(TrainingSpecGLDA<T, R>::make(0.5)) {
    }

    Tree(
      TreeNodePtr<T, R>     root,
      TrainingSpecPtr<T, R> training_spec) :
      root(std::move(root)),
      training_spec(std::move(training_spec)) {
    }

    Tree<T, R> retrain(stats::Data<T> &x, stats::DataColumn<R> &y) const {
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

    bool operator==(const Tree<T, R> &other) const {
      return *root == *other.root;
    }

    bool operator!=(const Tree<T, R> &other) const {
      return !(*this == other);
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
