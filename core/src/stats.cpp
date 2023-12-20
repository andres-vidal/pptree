#include "stats.hpp"
#include <vector>

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
Data<T> select_groups(
  Data<T>       data,
  DataColumn<G> data_groups,
  std::set<G>   groups) {
  std::vector<G> index;

  for (G i = 0; i < data_groups.rows(); i++) {
    if (groups.contains(data_groups(i))) {
      index.push_back(i);
    }
  }

  if (index.size() == 0) {
    return Data<T>(0, 0);
  }

  return data(index, Eigen::all);
}

template Data<double> select_groups<double, int>(
  Data<double>    data,
  DataColumn<int> data_groups,
  std::set<int>   groups);

template<typename T, typename G>
Data<T> remove_group(
  Data<T>       data,
  DataColumn<G> groups,
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
  int             group);

template<typename T, typename G>
DataColumn<G> binary_regroup(
  Data<T>       data,
  DataColumn<G> data_groups,
  std::set<G>   unique_groups) {
  struct Group {
    G indx;
    T mean;
    T diff;
  };

  auto cmp_indx_ascending = [](Group a, Group b) {
      return a.indx < b.indx;
    };

  auto cmp_mean_ascending = [](Group a, Group b) {
      return a.mean < b.mean;
    };

  auto cmp_diff_ascending = [](Group a, Group b) {
      return a.diff < b.diff;
    };

  std::vector<Group> wrappers(unique_groups.size());

  for (G g = 0; g < wrappers.size(); g++) {
    wrappers[g].indx = g;
    wrappers[g].mean = mean(select_group(data, data_groups, g))(0, 0);
  }

  std::sort(wrappers.begin(), wrappers.end(), cmp_mean_ascending);

  for (G g = 0; g < wrappers.size(); g++) {
    if (g ==  wrappers.size() - 1) {
      wrappers[g].diff = 0;
    } else {
      wrappers[g].diff = wrappers[g + 1].mean - wrappers[g].mean;
    }
  }

  Group edge_group = *std::max_element(wrappers.begin(), wrappers.end(), cmp_diff_ascending);
  std::sort(wrappers.begin(), wrappers.end(), cmp_indx_ascending);

  DataColumn<G> new_data_groups(data_groups.rows());

  for (int i = 0; i < new_data_groups.rows(); i++) {
    Group group = wrappers[data_groups(i)];

    if (group.mean <= edge_group.mean) {
      new_data_groups(i) = 0;
    } else {
      new_data_groups(i) = 1;
    }
  }

  return new_data_groups;
}

template DataColumn<int> binary_regroup<double, int>(
  Data<double>    data,
  DataColumn<int> data_groups,
  std::set<int>   unique_groups);

template<typename N>
std::set<N> unique(DataColumn<N> column) {
  std::set<N> unique_values;

  for (int i = 0; i < column.rows(); i++) {
    unique_values.insert(column(i));
  }

  return unique_values;
}

template std::set<int> unique<int>(DataColumn<int> column);

template<typename T, typename G>
Data<T> between_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups
  ) {
  DataColumn<T> global_mean = mean(data);
  Data<T> result = Data<T>::Zero(data.cols(), data.cols());

  for (G g = 0; g < unique_groups.size(); g++) {
    Data<T> group_data = select_group(data, groups, g);
    DataColumn<T> group_mean = mean(group_data);

    result += group_data.rows() * outer_square(group_mean - global_mean);
  }

  return result;
}

template Data<double> between_groups_sum_of_squares<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  std::set<int>   unique_groups);


template<typename T, typename G>
Data<T> within_groups_sum_of_squares(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups
  ) {
  Data<T> result = Data<T>::Zero(data.cols(), data.cols());

  for (G g = 0; g < unique_groups.size(); g++) {
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
  std::set<int>   unique_groups);
};
