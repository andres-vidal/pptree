#include "linalg.hpp"
#include <set>
#include <map>


namespace stats {
  template<typename T>
  using Data = linalg::DMatrix<T>;

  template<typename T>
  using DataColumn = linalg::DVector<T>;

  template<typename T, typename G>
  Data<T> select_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  Data<T> select_groups(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  groups);

  template<typename T, typename G>
  Data<T> remove_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  std::tuple<DataColumn<G>, std::set<int>, std::map<int, std::set<G> > >binary_regroup(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);

  template<typename N>
  std::set<N> unique(const DataColumn<N> &column);

  template<typename T, typename G>
  Data<T> between_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);


  template<typename T, typename G>
  Data<T> within_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);
}
