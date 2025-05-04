#pragma once

#include "DataSpec.hpp"
#include "GroupSpec.hpp"

#include "Invariant.hpp"
#include "Map.hpp"
#include "Normal.hpp"

#include "Logger.hpp"

#include <optional>

namespace models::stats {
  template<typename T, typename G>
  struct SortedDataSpec : public DataSpec<T, G> {
    private:

      static DataSpec<T, G> sort(const DataSpec<T, G> &data) {
        Data<T> x       = data.x;
        DataColumn<G> y = data.y;

        models::stats::sort(x, y);

        return DataSpec<T, G>(x, y, data.classes);
      }

      SortedDataSpec<T, G>(
        const Data<T>&       x,
        const DataColumn<G>& y,
        const std::set<G>&   classes,
        bool                 already_sorted) :
        DataSpec<T, G>(x, y, classes),
        group_spec(GroupSpec<T, G>(this->x, this->y)) {
      }

      SortedDataSpec<T, G>(
        const Data<T>&       x,
        const DataColumn<G>& y,
        bool                 already_sorted) :
        DataSpec<T, G>(x, y),
        group_spec(GroupSpec<T, G>(this->x, this->y)) {
      }

      SortedDataSpec<T, G>(
        const Data<T>&         x,
        const DataColumn<G>&   y,
        const std::set<G>&     classes,
        const GroupSpec<T, G>& group_spec) :
        DataSpec<T, G>(x, y, classes),
        group_spec(group_spec) {
      }

    public:

      const models::stats::GroupSpec<T, G> group_spec;

      SortedDataSpec() :
        DataSpec<T, G>(),
        group_spec(GroupSpec<T, G>(this->x, this->y)) {
      }

      SortedDataSpec(
        const Data<T> &       x,
        const DataColumn<G> & y,
        const std::set<G> &   classes) :
        DataSpec<T, G>(SortedDataSpec<T, G>::sort(DataSpec<T, G>(x, y, classes))),
        group_spec(GroupSpec<T, G>(this->x, this->y)) {
      }

      SortedDataSpec(
        const Data<T> &       x,
        const DataColumn<G> & y)
        : SortedDataSpec<T, G>(x, y, unique(y)) {
      }

      int group_size(const G &group) const {
        return group_spec.group_size(group);
      }

      int group_start(const G &group) const {
        return group_spec.group_start(group);
      }

      SortedDataSpec<T, G> select(const std::vector<int> &indices) const {
        return SortedDataSpec<T, G>(this->group_spec.data()(indices, Eigen::all), this->y(indices, Eigen::all), true);
      }

      SortedDataSpec<T, G> select(const std::set<int> &indices) const {
        return select(std::vector<int>(indices.begin(), indices.end()));
      }

      SortedDataSpec<T, G> get() const {
        return *this;
      }
  };

  struct SimulationParams {
    float mean            = 100.0f;
    float mean_separation = 50.0f;
    float sd              = 10.0f;
  };

  inline SortedDataSpec<float, int> simulate(
    const int               n,
    const int               p,
    const int               G,
    const SimulationParams& params = SimulationParams{}) {
    Data<float> x(n, p);
    DataColumn<int> y(n);

    for (int i = 0; i < n; ++i) {
      float group_mean = params.mean + (i % G) * params.mean_separation;

      Normal norm(group_mean, params.sd);

      for (int j = 0; j < p; ++j) {
        x(i, j) = norm();
      }

      y[i] = i % G;
    }

    return SortedDataSpec<float, int>(x, y);
  }

  std::pair<SortedDataSpec<float, int>, SortedDataSpec<float, int> >
  inline split(const SortedDataSpec<float, int>& data, float train_ratio) {
    const int n          = data.x.rows();
    const int train_size = static_cast<int>(n * train_ratio);

    std::vector<int> train_indices;
    std::vector<int> test_indices;
    train_indices.reserve(train_size);
    test_indices.reserve(n - train_size);

    LOG_INFO << "Splitting data of size " << n << " into " << train_size << " training and " << n - train_size << " test samples:";
    LOG_INFO << "Classes: " << data.classes << std::endl;
    LOG_INFO << "X: " << std::endl << data.x << std::endl;
    LOG_INFO << "Y: " << std::endl << data.y << std::endl;

    for (const auto& group : data.classes) {
      int group_start      = data.group_start(group);
      int group_size       = data.group_size(group);
      int group_end        = group_start + group_size - 1;
      int group_train_size = static_cast<int>(group_size * train_ratio);

      LOG_INFO << "Group " << group
               << ": start=" << group_start
               << ": end=" << group_end
               << ", size=" << group_size
               << ", train_size=" << group_train_size << std::endl;

      Uniform unif(group_start, group_end);
      std::vector<int> group_indices = unif.distinct(group_size);

      train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
      test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
    }

    LOG_INFO << "Final split has " << train_indices.size() << " training and " << test_indices.size() << " test samples:";

    return {
      SortedDataSpec<float, int>(data.x(train_indices, Eigen::all), data.y(train_indices)),
      SortedDataSpec<float, int>(data.x(test_indices, Eigen::all), data.y(test_indices))
    };
  }
}
