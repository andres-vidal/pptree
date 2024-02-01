#include "stats.hpp"
#include <vector>

#include <iostream>

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
      if (groups.find(data_groups(i)) != groups.end()) {
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
  struct Group {
    G id;
    T mean;
    T diff;
  };

  template<typename T, typename G>
  std::vector<Group<T, G> > summarize_groups(
    Data<T>       data,
    DataColumn<G> data_groups,
    std::set<G>   unique_groups) {
    std::vector<Group<T, G> > groups(unique_groups.size());
    int i = 0;

    for ( auto g : unique_groups) {
      groups[i].id = g;
      groups[i].mean = mean(select_group(data, data_groups, g)).value();
      i++;
    }

    return groups;
  }

  template<typename T, typename G>
  Group<T, G>get_edge_group(
    std::vector<Group<T, G> > groups) {
    auto cmp_mean_ascending = [](Group<T, G> a, Group<T, G> b) {
        return a.mean < b.mean;
      };

    auto cmp_diff_ascending = [](Group<T, G> a, Group<T, G> b) {
        return a.diff < b.diff;
      };

    std::sort(groups.begin(), groups.end(), cmp_mean_ascending);

    for (G g = 0; g < groups.size(); g++) {
      if (g ==  groups.size() - 1) {
        groups[g].diff = 0;
      } else {
        groups[g].diff = groups[g + 1].mean - groups[g].mean;
      }
    }

    return *std::max_element(groups.begin(), groups.end(), cmp_diff_ascending);
  }

  template<typename T, typename G>
  Group<T, G> get_group_by_id(
    std::vector<Group<T, G> > groups,
    G                         id) {
    auto matches_id = [id](Group<T, G> g) {
        return g.id == id;
      };

    return *std::find_if(groups.begin(), groups.end(), matches_id);
  }

  template<typename T, typename G>
  std::tuple<DataColumn<G>, std::set<int>, std::map<int, std::set<G> > > binary_regroup(
    Data<T>       data,
    DataColumn<G> data_groups,
    std::set<G>   unique_groups) {
    if (unique_groups.size() <= 2) {
      throw std::invalid_argument("Must have more than 2 groups to binary regroup");
    }

    if (data.cols() != 1) {
      throw std::invalid_argument("Data must be unidimensional to binary regroup");
    }

    std::vector<Group<T, G> > groups = summarize_groups(data, data_groups, unique_groups);
    Group<T, G> edge_group = get_edge_group(groups);

    DataColumn<G> new_data_groups(data_groups.rows());

    std::map<int, std::set<G> > group_mapping;

    for (int i = 0; i < new_data_groups.rows(); i++) {
      Group group = get_group_by_id(groups, data_groups(i));

      if (group.mean <= edge_group.mean) {
        new_data_groups(i) = 0;
        group_mapping[0].insert(group.id);
      } else {
        new_data_groups(i) = 1;
        group_mapping[1].insert(group.id);
      }
    }

    std::set<G> new_unique_groups = { 0, 1 };

    return std::make_tuple(new_data_groups, new_unique_groups, group_mapping);
  }

  template std::tuple < DataColumn<int>, std::set<int>, std::map<int, std::set<int> > > binary_regroup<double, int>(
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
