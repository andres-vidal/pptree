#pragma once

#include "DataSpec.hpp"

#include "Invariant.hpp"
namespace models::stats {
  template<typename T, typename G>
  struct SortedDataSpec : DataSpec<T, G> {
    private:
      template<typename GG>
      struct GroupSpec {
        int index;
        GroupSpec<GG> *next = nullptr;
        GroupSpec<GG> *prev = nullptr;
      };

      const std::map<G, GroupSpec<G> > group_specs;

      SortedDataSpec<T, G>(
        const Data<T> &                   x,
        const DataColumn<G> &             y,
        const std::set<G> &               classes,
        const std::map<G, GroupSpec<G> > &group_specs)
        : DataSpec<T, G>(x, y, classes),
        group_specs(group_specs) {
      }

      std::map<G,  GroupSpec<G> > init_group_specs() {
        std::map<G, GroupSpec<G> > specs;

        for (int i = 0; i < this->y.rows(); i++) {
          if (specs.count(this->y(i)) == 0) {
            G curr = this->y(i);

            specs[curr] = GroupSpec<G>{ i };

            if (i != 0) {
              G prev = this->y(i - 1);
              specs[curr].prev = &specs[prev];
              specs[prev].next = &specs[curr];
            }
          }
        }

        return specs;
      }

    public:

      SortedDataSpec(
        const Data<T> &       x,
        const DataColumn<G> & y,
        const std::set<G> &   classes)
        : DataSpec<T, G>(sort(DataSpec<T, G>(x, y, classes))),
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
        if (group_specs.at(group).next == nullptr) {
          return this->y.rows() - 1;
        }

        return group_specs.at(group).next->index - 1;
      }

      Data<T> group(const G &group) const {
        return this->x(Eigen::seq(group_start(group), group_end(group)), Eigen::all);
      }

      SortedDataSpec<T, G> analog(const Data<T> &data) const {
        return SortedDataSpec<T, G>(data, this->y, this->classes, this->group_specs);
      }
  };

  template<typename T, typename G>
  DataSpec<T, G> sort(const DataSpec<T, G> &data) {
    std::vector<int> indices(data.x.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [&data](int idx1, int idx2) {
       return data.y(idx1) < data.y(idx2);
     });

    return DataSpec<T, G>(
      select_rows(data.x, indices),
      select_rows(data.y, indices),
      data.classes);
  }
}
