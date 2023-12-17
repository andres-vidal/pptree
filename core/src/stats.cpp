#include "stats.hpp"

using namespace linalg;

namespace stats {
template<typename T, typename G>
Data<T> select_group(
  Data<T>       data,
  DataColumn<G> groups,
  G             group
  ) {
  std::vector<G> index;

  for (G i = 0; i < groups.rows(); i++) {
    if (groups(i) == group) {
      index.push_back(i);
    }
  }

  if (index.size() == 0) {
    return Data<T>(0, 0);
  }

  return data(index, Eigen::all);
}

template Data<double> select_group<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  int             group);


template<typename T, typename G>
Data<T> remove_group(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count,
  G             group
  ) {
  std::vector<G> index;

  for (G i = 0; i < groups.rows(); i++) {
    if (groups(i) != group) {
      index.push_back(i);
    }
  }

  if (index.size() == 0) {
    return Data<T>(0, 0);
  }

  return data(index, Eigen::all);
}

template Data<double> remove_group<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  int             group_count,
  int             group);


template<typename T, typename G>
Data<T> between_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count
  ) {
  DataColumn<T> global_mean = mean(data);
  Data<T> result = Data<T>::Zero(data.cols(), data.cols());

  for (G g = 0; g < group_count; g++) {
    Data<T> group_data = select_group(data, groups, g);
    DataColumn<T> group_mean = mean(group_data);

    result += group_data.rows() * outer_square(group_mean - global_mean);
  }

  return result;
}

template Data<double> between_groups_sum_of_squares<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  int             group_count);


template<typename T, typename G>
Data<T> within_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  int           group_count
  ) {
  Data<T> result = Data<T>::Zero(data.cols(), data.cols());

  for (G g = 0; g < group_count; g++) {
    Data<T> group_data = select_group(data, groups, g);
    DataColumn<T> group_mean = mean(group_data);
    Data<T> centered_data = group_data.rowwise() - group_mean.transpose();

    for (int r = 0; r < centered_data.rows(); r++) {
      result += outer_square(centered_data.row(r));
    }
  }

  return result;
}

template Data<double> within_groups_sum_of_squares<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  int             group_count);
};
