#include "linalg.hpp"

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
Data<T> remove_group(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count,
  G             group);

template<typename T, typename G>
Data<T> between_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count);


template<typename T, typename G>
Data<T> within_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count);
}
