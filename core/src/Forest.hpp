#pragma once

#include "BootstrapDataSpec.hpp"
#include "Tree.hpp"

template<typename T, typename R>
using BootstrapTree = Tree<T, R, BootstrapDataSpec<T, R> >;


template<typename T, typename R>
struct Forest {
  std::vector<std::unique_ptr<BootstrapTree<T, R> > > trees;
  std::unique_ptr<TrainingSpec<T, R> > training_spec;
  std::shared_ptr<DataSpec<T, R> > training_data;
  const double seed = 0.0;

  Forest() {
  }

  Forest(
    std::unique_ptr<TrainingSpec<T, R> > && training_spec,
    std::shared_ptr<DataSpec<T, R> > &&     training_data,
    const double                            seed)
    : training_spec(std::move(training_spec)),
    training_data(training_data),
    seed(seed) {
  }

  R predict(const DataColumn<T> &data) const {
    std::map<R, int> votes_per_group;

    for (const auto &tree : trees) {
      R prediction = tree->predict(data);

      if (votes_per_group.find(prediction) == votes_per_group.end()) {
        votes_per_group[prediction] = 1;
      } else {
        votes_per_group[prediction] += 1;
      }
    }

    int most_voted_group_votes = 0;
    R most_voted_group;

    for (const auto &[key, votes] : votes_per_group) {
      if (votes > most_voted_group_votes) {
        most_voted_group = key;
        most_voted_group_votes = votes;
      }
    }

    return most_voted_group;
  }

  DataColumn<R> predict(const Data<T> &data) const {
    DataColumn<R> predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((DataColumn<T>)data.row(i));
    }

    return predictions;
  }

  void add_tree(std::unique_ptr<BootstrapTree<T, R> > tree) {
    trees.push_back(std::move(tree));
  }

  bool operator==(const Forest<T, R> &other) const {
    if (trees.size() != other.trees.size()) {
      return false;
    }

    for (int i = 0; i < trees.size(); i++) {
      if (*trees[i] != *other.trees[i]) {
        return false;
      }
    }

    return true;
  }

  bool operator!=(const Forest<T, R> &other) const {
    return !(*this == other);
  }

  Forest<T, R> retrain(const DataSpec<T, R> &data) const {
    return train(
      *training_spec,
      data,
      trees.size(),
      seed);
  }

  Projector<T> variable_importance() const {
    struct TreeImportance {
      Projector<T> operator()(Projector<T> acc, const std::unique_ptr<BootstrapTree<T, R> >& tree) {
        return acc + tree->variable_importance();
      }
    };

    Forest<T, R> std_forest = retrain(center(descale(*training_data)));

    Projector<T> importance = std::accumulate(
      std_forest.trees.begin(),
      std_forest.trees.end(),
      Projector<T>(Projector<T>::Zero(training_data->x.cols())),
      TreeImportance());


    return importance.array() / std_forest.trees.size();
  }
};

template<typename T, typename R >
Forest<T, R> train(
  const TrainingSpec<T, R> &training_spec,
  const DataSpec<T, R> &    training_data,
  const int                 size,
  const double              seed);
