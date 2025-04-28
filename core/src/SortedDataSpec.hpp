#pragma once

#include "DataSpec.hpp"

#include "Invariant.hpp"
#include "Map.hpp"
#include "Normal.hpp"

#include "Logger.hpp"

#include <optional>

namespace models::stats {
  template<typename T, typename G>
  struct SortedDataSpec : public DataSpec<T, G> {
    private:

      struct GroupSpec {
        int index;
        std::optional<G> next;
        std::optional<G> prev;
      };

      const std::map<G, GroupSpec > group_specs;

      SortedDataSpec<T, G>(
        const Data<T> &               x,
        const DataColumn<G> &         y,
        const std::set<G> &           classes,
        const std::map<G, GroupSpec> &group_specs)
        : DataSpec<T, G>(x, y, classes),
        group_specs(group_specs) {
      }

      std::map<G,  GroupSpec > init_group_specs() {
        std::map<G, GroupSpec > specs;

        for (int i = 0; i < this->y.rows(); i++) {
          if (specs.count(this->y(i)) == 0) {
            G curr = this->y(i);

            specs[curr] = GroupSpec{ i };

            if (i != 0) {
              G prev = this->y(i - 1);
              specs[curr].prev = prev;
              specs[prev].next = curr;
            }
          }
        }

        return specs;
      }

      static DataSpec<T, G> sort(const DataSpec<T, G> &data) {
        std::vector<int> indices(data.x.rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::stable_sort(indices.begin(), indices.end(), [&data](int idx1, int idx2) {
            return data.y(idx1) < data.y(idx2);
          });

        return DataSpec<T, G>(
          data.x(indices, Eigen::all),
          data.y(indices, Eigen::all),
          data.classes);
      }

    public:

      SortedDataSpec(
        const Data<T> &       x,
        const DataColumn<G> & y,
        const std::set<G> &   classes)
        : DataSpec<T, G>(SortedDataSpec<T, G>::sort(DataSpec<T, G>(x, y, classes))),
        group_specs(init_group_specs()) {
      }

      SortedDataSpec(
        const Data<T> &       x,
        const DataColumn<G> & y)
        : SortedDataSpec<T, G>(x, y, unique(y)) {
      }

      //cppcheck-suppress noExplicitConstructor
      SortedDataSpec(
        const DataSpec<T, G> &data)
        : SortedDataSpec<T, G>(data.x, data.y, data.classes) {
      }

      int group_size(const G &group) const {
        return 1 + group_end(group) - group_start(group);
      }

      int group_start(const G &group) const {
        return group_specs.at(group).index;
      }

      int group_end(const G &group) const {
        std::optional<G> next = group_specs.at(group).next;

        if (!next.has_value()) {
          return this->y.rows() - 1;
        }

        return group_specs.at(next.value()).index - 1;
      }

      DataView<T> group(const G &group) const {
        return this->x(Eigen::seq(group_start(group), group_end(group)), Eigen::all);
      }

      SortedDataSpec<T, G> analog(const Data<T> &data) const {
        return SortedDataSpec<T, G>(data, this->y, this->classes, this->group_specs);
      }

      SortedDataSpec<T, G> remap(const std::map<G, G> &mapping) const {
        std::vector<G> sorted_old_groups = utils::sort_keys_by_value(mapping);

        Data<T> new_x(this->x.rows(), this->x.cols());
        DataColumn<G> new_y(this->y.rows());
        std::set<G> new_classes;
        std::map<G, GroupSpec> new_group_specs;

        int batch_start = 0;

        for (int i = 0; i < sorted_old_groups.size(); i++) {
          G old_group = sorted_old_groups[i];
          G new_group = mapping.at(old_group);

          int batch_end = batch_start + group_size(old_group) - 1;

          new_x(Eigen::seq(batch_start, batch_end), Eigen::all) = group(old_group);
          new_y(Eigen::seq(batch_start, batch_end)).setConstant(new_group);
          new_classes.insert(new_group);

          if (!new_group_specs.count(new_group)) {
            new_group_specs[new_group] = GroupSpec{ batch_start };

            if (i != 0) {
              G prev_new_group = mapping.at(sorted_old_groups[i - 1]);
              new_group_specs[new_group].prev      = prev_new_group;
              new_group_specs[prev_new_group].next = new_group;
            }
          }

          batch_start = batch_end + 1;
        }

        return SortedDataSpec<T, G>(
          new_x,
          new_y,
          new_classes,
          new_group_specs);
      }

      SortedDataSpec<T, G> subset(const std::set<G> &groups) const {
        int subset_size = 0;

        for (const G &g : groups) {
          subset_size += group_size(g);
        }

        Data<T> new_x(subset_size, this->x.cols());
        DataColumn<G> new_y(subset_size);
        std::set<G> new_classes = groups;
        std::map<G, GroupSpec> new_group_specs;

        int batch_start = 0;

        for (const G &g : groups) {
          int batch_end = batch_start + group_size(g) - 1;

          new_x(Eigen::seq(batch_start, batch_end), Eigen::all) = group(g);
          new_y(Eigen::seq(batch_start, batch_end)).setConstant(g);
          new_classes.insert(g);

          if (!new_group_specs.count(g)) {
            new_group_specs[g] = GroupSpec{ batch_start };

            if (batch_start != 0) {
              G prev_g = new_y(batch_start - 1);
              new_group_specs[g].prev      = prev_g;
              new_group_specs[prev_g].next = g;
            }
          }

          batch_start = batch_end + 1;
        }

        return SortedDataSpec<T, G>(
          new_x,
          new_y,
          new_classes,
          new_group_specs);
      }

      virtual Data<T> bgss() const {
        DataColumn<T> global_mean = mean(this->x);
        Data<T> result            = Data<T>::Zero(this->x.cols(), this->x.cols());

        for (const G &g : this->classes) {
          DataView<T> group_data      = group(g);
          DataColumn<T> group_mean    = mean(group_data);
          DataColumn<T> centered_mean = group_mean - global_mean;

          result.noalias() += group_data.rows() * math::outer_square(centered_mean);
        }

        return result;
      }

      virtual Data<T> wgss() const {
        Data<T> result = Data<T>::Zero(this->x.cols(), this->x.cols());

        for (const G &g : this->classes) {
          DataView<T> group_data            = group(g);
          DataColumn<T> centered_group_data = center(group_data);

          result.noalias() += math::inner_square(centered_group_data);
        }

        return result;
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
