#pragma once

#include "BootstrapDataSpec.hpp"
#include "Tree.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace models {
  template<typename T, typename R>
  using BootstrapTree = Tree<T, R, stats::BootstrapDataSpec<T, R> >;

  template<typename T, typename R>
  struct Forest {
    std::vector<std::unique_ptr<BootstrapTree<T, R> > > trees;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;
    std::shared_ptr<stats::DataSpec<T, R> > training_data;
    const double seed = 0.0;

    Forest() {
    }

    Forest(
      std::unique_ptr<TrainingSpec<T, R> > &&    training_spec,
      std::shared_ptr<stats::DataSpec<T, R> > && training_data,
      const double                               seed)
      : training_spec(std::move(training_spec)),
      training_data(training_data),
      seed(seed) {
    }

    R predict(const stats::DataColumn<T> &data) const {
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

    stats::DataColumn<R> predict(const stats::Data<T> &data) const {
      stats::DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((stats::DataColumn<T>)data.row(i));
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

    Forest<T, R> retrain(const stats::DataSpec<T, R> &data) const {
      return train(
        *training_spec,
        data,
        trees.size(),
        seed);
    }

    pp::Projector<T> variable_importance() const {
      struct TreeImportance {
        pp::Projector<T> operator()(pp::Projector<T> acc, const std::unique_ptr<BootstrapTree<T, R> >& tree) {
          return acc + tree->variable_importance();
        }
      };

      Forest<T, R> std_forest = retrain(center(descale(*training_data)));

      pp::Projector<T> importance = std::accumulate(
        std_forest.trees.begin(),
        std_forest.trees.end(),
        pp::Projector<T>(pp::Projector<T>::Zero(training_data->x.cols())),
        TreeImportance());


      return importance.array() / std_forest.trees.size();
    }
  };

  template<typename T, typename R >
  Forest<T, R> train(
    const TrainingSpec<T, R> &            training_spec,
    const models::stats::DataSpec<T, R> & training_data,
    const int                             size,
    const double                          seed);


  template<typename T, typename R>
  void to_json(json& j, const Forest<T, R>& forest) {
    std::vector<json> trees_json;

    for (const auto& tree : forest.trees) {
      json tree_json;
      to_json(tree_json, *tree);
      trees_json.push_back(tree_json);
    }

    j = json{ { "trees", trees_json } };
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Forest<T, R>& forest) {
    json json_response(forest);
    return ostream << json_response.dump(2, ' ', false);
  }
}
