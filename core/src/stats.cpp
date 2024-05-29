#include "stats.hpp"

using namespace linalg;

namespace stats {
  template<typename T>
  Data<T> select_rows(
    const Data<T> &         data,
    const std::vector<int> &indices) {
    Data<T> result(indices.size(), data.cols());

    for (int i = 0; i < indices.size(); i++) {
      result.row(i) = data.row(indices[i]);
    }

    return result;
  }

  template Data<long double> select_rows<long double>(
    const Data<long double> & data,
    const std::vector<int> &  indices);

  template<typename T>
  Data<T> select_rows(
    const Data<T> &      data,
    const std::set<int> &indices) {
    return select_rows(data, std::vector<int>(indices.begin(), indices.end()));
  }

  template Data<long double> select_rows<long double>(
    const Data<long double> & data,
    const std::set<int> &     indices);

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> &   data,
    const std::vector<int> &indices) {
    return select_rows((Data<T>)data, indices).col(0);
  }

  template DataColumn<int> select_rows<int>(
    const DataColumn<int> &  data,
    const std::vector<int> & indices);

  template DataColumn<long double> select_rows<long double>(
    const DataColumn<long double> & data,
    const std::vector<int> &        indices);

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> & data,
    const std::set<int> & indices) {
    return select_rows((Data<T>)data, indices).col(0);
  }

  template DataColumn<int> select_rows<int>(
    const DataColumn<int> & data,
    const std::set<int> &   indices);

  template DataColumn<long double> select_rows<long double>(
    const DataColumn<long double> & data,
    const std::set<int> &           indices);

  template<typename G>
  std::vector<G> select_group(
    const DataColumn<G> &groups,
    const G &            group) {
    std::vector<G> indices;

    for (int i = 0; i < groups.rows(); i++) {
      if (groups(i) == group) {
        indices.push_back(i);
      }
    }

    return indices;
  }

  template<typename T, typename G>
  Data<T> select_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group
    ) {
    std::vector<G> indices = select_group(groups, group);

    if (indices.size() == 0) {
      return Data<T>(0, 0);
    }

    return data(indices, Eigen::all);
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
      Data<T> centered_group_data = center(select_group(data, groups, group));

      for (int r = 0; r < centered_group_data.rows(); r++) {
        result += outer_square(centered_group_data.row(r));
      }
    }

    return result;
  }

  template Data<long double> within_groups_sum_of_squares<long double, int>(
    const Data<long double> & data,
    const DataColumn<int>&    groups,
    const std::set<int> &     unique_groups);

  template<typename T, typename G>
  BootstrapDataSpec<T, G> stratified_proportional_sample(
    const DataSpec<T, G> &data,
    const int             size,
    std::mt19937 &        rng) {
    assert(size > 0 && "Sample size must be greater than 0.");
    assert(size <= data.y.rows() && "Sample size cannot be larger than the number of rows in the data.");

    const int data_size = data.y.rows();

    std::vector<int> sample_indices;

    for (const G& group : data.classes) {
      const std::vector<int> group_indices = select_group(data.y, group);

      const int group_size = group_indices.size();
      const int group_sample_size = std::round(group_size / (double)data_size * size);

      for (int i = 0; i < group_sample_size; i++) {
        const Uniform unif(0, group_indices.size() - 1);
        const int sampled_index = group_indices[unif(rng)];
        sample_indices.push_back(sampled_index);
      }
    }

    return BootstrapDataSpec<T, G>(data.x, data.y, data.classes, sample_indices);
  }

  template BootstrapDataSpec<long double, int> stratified_proportional_sample(
    const DataSpec<long double, int>& data,
    const int                         size,
    std::mt19937 &                    rng);

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


  template<typename T>
  DataColumn<T> mean(
    const Data<T> &data
    ) {
    return data.colwise().mean();
  }

  template DataColumn<long double> mean<long double>(
    const Data<long double> &data);

  template<typename T>
  T mean(
    const DataColumn<T> &data
    ) {
    return data.mean();
  }

  template long double mean(
    const DataColumn<long double> &data);

  template<typename T>
  Data<T> covariance(
    const Data<T> &data
    ) {
    Data<T> centered = center(data);

    return (centered.transpose() * centered) / (data.rows() - 1);
  }

  template Data<long double> covariance<long double>(
    const Data<long double> &data);

  template<typename T>
  DataColumn<T> sd(
    const Data<T> &data
    ) {
    return covariance(data).diagonal().array().sqrt();
  }

  template DataColumn<long double> sd<long double>(
    const Data<long double> &data);


  template<typename T>
  T sd(
    const DataColumn<T> &data) {
    return sqrt((inner_square(center(data))) / (data.rows() - 1));
  }

  template long double sd(
    const DataColumn<long double> &data);

  template<typename T>
  Data<T> center(
    const Data<T> &data) {
    return data.rowwise() - mean(data).transpose();
  }

  template Data<long double> center<long double>(
    const Data<long double> &data);

  template<typename T>
  DataColumn<T> center(
    const DataColumn<T> &data) {
    return data.array() - mean(data);
  }

  template DataColumn<long double> center<long double>(
    const DataColumn<long double> &data);

  template<typename T, typename R>
  DataSpec<T, R> center(
    const DataSpec<T, R> &data) {
    return DataSpec<T, R>(center(data.x), data.y, data.classes);
  }

  template DataSpec<long double, int> center<long double, int>(
    const DataSpec<long double, int> &data);

  template<typename T, typename G>
  BootstrapDataSpec<T, G> center(
    const BootstrapDataSpec<T, G> &data) {
    return BootstrapDataSpec<T, G>(center(data.x), data.y, data.classes, data.sample_indices);
  }

  template BootstrapDataSpec<long double, int> center<long double, int>(
    const BootstrapDataSpec<long double, int> &data);

  template<typename T>
  Data<T> descale(
    const Data<T> &data) {
    DataColumn<T> scaling_factor = sd(data);

    for (int i = 0; i < scaling_factor.rows(); i++) {
      if (scaling_factor(i) == 0) {
        scaling_factor(i) = 1;
      }
    }

    return data.array().rowwise() / scaling_factor.transpose().array();
  }

  template Data<long double> descale<long double>(
    const Data<long double> &data);

  template<typename T>
  DataColumn<T> descale(
    const DataColumn<T> &data) {
    T scaling_factor = sd(data);

    if (scaling_factor == 0) {
      scaling_factor = 1;
    }

    return data.array() / scaling_factor;
  }

  template DataColumn<long double> descale<long double>(
    const DataColumn<long double> &data);


  template<typename T, typename R>
  DataSpec<T, R> descale(
    const DataSpec<T, R> &data) {
    return DataSpec<T, R>(descale(data.x), data.y, data.classes);
  }

  template DataSpec<long double, int> descale<long double, int>(
    const DataSpec<long double, int> &data);

  template<typename T, typename R>
  BootstrapDataSpec<T, R> descale(
    const BootstrapDataSpec<T, R> &data) {
    return BootstrapDataSpec<T, R>(descale(data.x), data.y, data.classes, data.sample_indices);
  }

  template BootstrapDataSpec<long double, int> descale<long double, int>(
    const BootstrapDataSpec<long double, int> &data);

  template<typename T>
  Data<T> shuffle_column(
    const Data<T> &data,
    const int      column,
    std::mt19937 & rng) {
    Data<T> shuffled = data;

    std::vector<int> indices(data.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < data.rows(); i++) {
      shuffled(i, column) = data(indices[i], column);
    }

    return shuffled;
  }

  template Data<long double> shuffle_column<long double>(
    const Data<long double> &data,
    const int                column,
    std::mt19937 &           rng);
};
