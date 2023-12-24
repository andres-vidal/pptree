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
  Data<T>       data,
  DataColumn<G> groups,
  G             group);

template<typename T, typename G>
Data<T> select_groups(
  Data<T>       data,
  DataColumn<G> data_groups,
  std::set<G>   groups);

template<typename T, typename G>
Data<T> remove_group(
  Data<T>       data,
  DataColumn<G> groups,
  G             group);

template<typename T, typename G>
std::tuple < DataColumn<G>, std::set<int>, std::map<int, std::set<G> > >binary_regroup(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups);

template<typename N>
std::set<N> unique(DataColumn<N> column);

template<typename T, typename G>
Data<T> between_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups);


template<typename T, typename G>
Data<T> within_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups);
}
