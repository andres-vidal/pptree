#include "stats.hpp"
#include <iostream>

using namespace linalg;

namespace stats {
  template<typename T, typename G>
  Data<T> select_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group
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

  template Data<long double> select_group<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const int &               group);

  template Data<int> select_group<int, int>(
    const Data<int> &      data,
    const DataColumn<int> &groups,
    const int &            group);

  template<typename T, typename G>
  DataColumn<T> select_group(
    const DataColumn<T> &data,
    const DataColumn<G> &groups,
    const G &            group) {
    return (DataColumn<T>)select_group((Data<T>)data, groups, group);
  }

  template DataColumn<long double> select_group<long double, int>(
    const DataColumn<long double> &data,
    const DataColumn<int> &        groups,
    const int &                    group);

  template DataColumn<int> select_group<int, int>(
    const DataColumn<int> &data,
    const DataColumn<int> &groups,
    const int &            group);

  template<typename T, typename G>
  Data<T> select_groups(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  groups) {
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

  template Data<long double> select_groups<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   data_groups,
    const std::set<int> &     groups);

  template<typename T, typename G>
  Data<T> remove_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group
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

  template Data<long double> remove_group<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const int &               group);


  template<typename T, typename G>
  struct Group {
    G id;
    T mean;
    T diff;
  };

  template<typename T, typename G>
  std::vector<Group<T, G> > summarize_groups(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  unique_groups) {
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
    const std::vector<Group<T, G> > &groups,
    const G &                        id) {
    auto matches_id = [id](Group<T, G> g) {
        return g.id == id;
      };

    return *std::find_if(groups.begin(), groups.end(), matches_id);
  }

  template<typename T, typename G>
  std::tuple<DataColumn<G>, std::set<int>, std::map<int, std::set<G> > > binary_regroup(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  unique_groups) {
    assert(unique_groups.size() > 2 && "Must have more than 2 groups to binary regroup");
    assert(data.cols() == 1 && "Data must be unidimensional to binary regroup");

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

    return { new_data_groups, new_unique_groups, group_mapping };
  }

  template std::tuple < DataColumn<int>, std::set<int>, std::map<int, std::set<int> > > binary_regroup<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   data_groups,
    const std::set<int> &     unique_groups);

  template<typename N>
  std::set<N> unique(const DataColumn<N> &column) {
    std::set<N> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  template std::set<int> unique<int>(const DataColumn<int> &column);

  template<typename T, typename G>
  Data<T> between_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups
    ) {
    DataColumn<T> global_mean = mean(data);
    Data<T> result = Data<T>::Zero(data.cols(), data.cols());

    for (const G& group : unique_groups) {
      Data<T> group_data = select_group(data, groups, group);
      DataColumn<T> group_mean = mean(group_data);

      result += group_data.rows() * outer_square(group_mean - global_mean);
    }

    return result;
  }

  template Data<long double> between_groups_sum_of_squares<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups);


  template<typename T, typename G>
  Data<T> within_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups
    ) {
    Data<T> result = Data<T>::Zero(data.cols(), data.cols());

    for (const G& group : unique_groups) {
      Data<T> group_data = select_group(data, groups, group);
      DataColumn<T> group_mean = mean(group_data);
      Data<T> centered_data = group_data.rowwise() - group_mean.transpose();

      for (int r = 0; r < centered_data.rows(); r++) {
        result += outer_square(centered_data.row(r));
      }
    }

    return result;
  }

  template Data<long double> within_groups_sum_of_squares<long double, int>(
    const Data<long double> & data,
    const DataColumn<int>&    groups,
    const std::set<int> &     unique_groups);


  template<typename T>
  Data<T> sample(const Data<T>& data, int size, std::mt19937 &rng) {
    assert(size > 0 && "Sample size must be greater than 0.");
    assert(size <= data.rows() && "Sample size cannot be larger than the number of rows in the data.");

    Data<T> result(size, data.cols());

    for (int i = 0; i < size; ++i) {
      Uniform unif(0, data.rows() - 1);
      int sampled_index = unif(rng);
      result.row(i) = data.row(sampled_index);
    }

    return result;
  }

  template Data<long double> sample<long double>(
    const Data<long double> &data,
    int                      size,
    std::mt19937 &           rng);


  template<typename T, typename G>
  std::tuple<Data<T>, DataColumn<G> > stratified_sample(
    const Data<T> &        data,
    const DataColumn<G> &  groups,
    const std::map<G, int> sizes,
    std::mt19937 &         rng) {
    int total_size = 0;

    for (const auto& [group, size] : sizes) {
      total_size += size;
    }

    Data<T> sample_data(total_size, data.cols());
    DataColumn<G> sample_groups(total_size);

    int acc = 0;

    for ( const auto& [group, size] : sizes) {
      Data<T> group_sample = stats::sample(select_group(data, groups, group), size, rng);

      for (int i = 0; i < size; i++) {
        sample_data.row(i + acc) = group_sample.row(i);
        sample_groups(i + acc) = group;
      }

      acc += size;
    }

    return { sample_data, sample_groups };
  }

  template std::tuple<Data<long double>, DataColumn<int> > stratified_sample(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::map<int, int>  sizes,
    std::mt19937 &            rng);

  template<typename T, typename G>
  std::tuple<Data<T>, DataColumn<G> > stratified_proportional_sample(
    const Data<T> &       data,
    const DataColumn<G> & groups,
    const std::set<G> &   unique_groups,
    const int             size,
    std::mt19937 &        rng) {
    assert(size > 0 && "Sample size must be greater than 0.");
    assert(size <= data.rows() && "Sample size cannot be larger than the number of rows in the data.");

    std::map<G, int> sizes;

    for (const G& group : unique_groups) {
      sizes[group] = round(size * select_group(data, groups, group).rows() / (double)data.rows());
    }

    return stratified_sample(data, groups, sizes, rng);
  }

  template std::tuple<Data<long double>, DataColumn<int> > stratified_proportional_sample(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups,
    const int                 size,
    std::mt19937 &            rng);


  template<typename T>
  std::tuple<std::vector<int>, std::vector<int> > mask_null_columns(const Data<T> &data) {
    std::vector<int> mask(data.cols());
    std::vector<int> index;

    for (int i = 0; i < data.cols(); i++) {
      if (data.col(i).minCoeff() == 0 && data.col(i).maxCoeff() == 0) {
        mask[i] = 0;
      } else {
        mask[i] = 1;
        index.push_back(i);
      }
    }

    return { mask, index };
  }

  template std::tuple<std::vector<int>, std::vector<int> > mask_null_columns<long double>(
    const Data<long double> &data);

  template<typename T>
  DataColumn<T> expand(
    const DataColumn<T> &   data,
    const std::vector<int> &mask) {
    DataColumn<T> expanded = DataColumn<T>::Zero(mask.size());

    int j = 0;

    for (int i = 0; i < mask.size(); i++) {
      if (mask[i] == 1) {
        expanded.row(i) = data.row(j);
        j++;
      }
    }

    return expanded;
  }

  template DataColumn<long double> expand<long double>(
    const DataColumn<long double> &data,
    const std::vector<int> &       mask);
};
