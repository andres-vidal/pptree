#include "Group.hpp"

#include <vector>
namespace pptree::stats {
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

  template std::tuple<DataColumn<int>, std::set<int>, std::map<int, std::set<int> > > binary_regroup(
    const Data<long double> & data,
    const DataColumn<int> &   data_groups,
    const std::set<int> &     unique_groups);
}
