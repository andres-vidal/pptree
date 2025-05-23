#pragma once

#include "BootstrapTree.hpp"

#include <nlohmann/json.hpp>
#include <set>
#include <algorithm>
#include <thread>

namespace models {
  using json = nlohmann::json;

  template <typename T, typename R>
  class VIStrategy;

  template<typename T, typename R>
  struct Forest {
    static Forest<T, R> train(
      const TrainingSpec<T, R> & training_spec,
      stats::Data<T> &           x,
      stats::DataColumn<R> &     y,
      const int                  size,
      const int                  seed,
      const int                  n_threads = std::thread::hardware_concurrency());


    std::vector<std::unique_ptr<BootstrapTree<T, R> > > trees;

    TrainingSpecPtr<T, R> training_spec;

    const stats::Data<T> x;
    const stats::DataColumn<R> y;
    const std::set<R> classes;

    const int seed      = 0;
    const int n_threads = 1;

    Forest() {
    }

    Forest(
      TrainingSpecPtr<T, R> &&     training_spec,
      const stats::Data<T> &       x,
      const stats::DataColumn<R> & y,
      const std::set<R> &          classes,
      const int                    seed)
      : training_spec(std::move(training_spec)),
      x(x),
      y(y),
      classes(classes),
      seed(seed),
      n_threads(std::thread::hardware_concurrency()) {
    }

    Forest(
      TrainingSpecPtr<T, R> &&     training_spec,
      const stats::Data<T> &       x,
      const stats::DataColumn<R> & y,
      const std::set<R> &          classes,
      const int                    seed,
      const int                    n_threads)
      : training_spec(std::move(training_spec)),
      x(x),
      y(y),
      classes(classes),
      seed(seed),
      n_threads(std::clamp(n_threads, 1, (int) std::thread::hardware_concurrency())) {
    }

    R predict(const stats::DataColumn<T> &data) const {
      std::vector<int> indx(trees.size());
      std::iota(indx.begin(), indx.end(), 0);
      return predict(data, indx);
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

      for (std::size_t i = 0; i < trees.size(); i++) {
        if (*trees[i] != *other.trees[i]) {
          return false;
        }
      }

      return true;
    }

    bool operator!=(const Forest<T, R> &other) const {
      return !(*this == other);
    }

    Forest<T, R> retrain(stats::Data<T> &x,  stats::DataColumn<R> &y) const {
      return Forest<T, R>::train(
        *training_spec,
        x,
        y,
        trees.size(),
        seed,
        n_threads);
    }

    double error_rate(const stats::Data<T> &x, const stats::DataColumn<R> &y) const {
      return stats::error_rate(predict(x), y);
    }

    double error_rate() const {
      std::set<int> oob_indices = get_oob_indices();
      std::vector<int> oob_indices_vec(oob_indices.begin(), oob_indices.end());
      stats::DataColumn<R> oob_predictions = oob_predict(oob_indices);
      stats::DataColumn<R> oob_y           = y(oob_indices_vec, Eigen::all);
      return stats::error_rate(oob_predictions, oob_y);
    }

    stats::ConfusionMatrix confusion_matrix(const stats::Data<T> &x, const stats::DataColumn<R> &y) const {
      return stats::ConfusionMatrix(predict(x), y);
    }

    stats::ConfusionMatrix confusion_matrix() const {
      std::set<int> oob_indices = get_oob_indices();
      std::vector<int> oob_indices_vec(oob_indices.begin(), oob_indices.end());
      stats::DataColumn<R> oob_predictions = oob_predict(oob_indices);
      stats::DataColumn<R> oob_y           = y(oob_indices_vec, Eigen::all);
      return stats::ConfusionMatrix(oob_predictions, oob_y);
    }

    math::DVector<T> variable_importance(const VIStrategy<T, R> &strategy) const {
      return strategy(*this);
    }

    json to_json() const {
      std::vector<json> trees_json;

      for (const auto& tree : trees) {
        trees_json.push_back(tree->to_json());
      }

      return json{
        { "trees", trees_json }
      };
    }

    private:

      R predict(const stats::DataColumn<T> data, const std::vector<int>&    indx) const {
        std::map<R, int> votes_per_group;

        for (const auto &i : indx) {
          R prediction = trees[i]->predict(data);

          if (votes_per_group.find(prediction) == votes_per_group.end()) {
            votes_per_group[prediction] = 1;
          } else {
            votes_per_group[prediction] += 1;
          }
        }

        int most_voted_group_votes = 0;
        R most_voted_group         = 0;

        for (const auto &[key, votes] : votes_per_group) {
          if (votes > most_voted_group_votes) {
            most_voted_group       = key;
            most_voted_group_votes = votes;
          }
        }

        return most_voted_group;
      }

      R oob_predict(int index) const {
        std::vector<int> tree_indices;

        for (int i = 0; i < trees.size(); i++) {
          bool is_oob = trees[i]->oob_indices.count(index);

          if (is_oob) {
            tree_indices.push_back(i);
          }
        }

        return predict(x.row(index), tree_indices);
      }

      stats::DataColumn<R> oob_predict(const std::set<int> &indices) const {
        stats::DataColumn<R> predictions(indices.size());

        int i = 0;

        for (int index : indices) {
          predictions(i) = oob_predict(index);
          i++;
        }

        return predictions;
      }

      std::set<int> get_oob_indices() const {
        std::set<int> indices;

        for (const auto& tree : trees) {
          std::set<int> oob_indices = tree->oob_indices;

          std::set<int> temp;
          std::set_union(
            indices.begin(), indices.end(),
            oob_indices.begin(), oob_indices.end(),
            std::inserter(temp, temp.begin()));

          indices = temp;
        }

        return indices;
      }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Forest<T, R>& forest) {
    return ostream << forest.to_json().dump(2, ' ', false);
  }
}
