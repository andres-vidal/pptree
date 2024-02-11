#include "pptree.hpp"
#include "pptreeio.hpp"
#include <vector>
#include <set>


using namespace pptree;
using namespace Eigen;

namespace pptree {
  template<typename T, typename R >
  std::tuple<DataColumn<R>, std::set<int>, std::map<int, std::set<R> > >as_binary_problem(
    Data<T>          data,
    DataColumn<R>    groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy);
  template<typename T, typename R >
  Threshold<T> get_threshold(
    DataColumn<T> projected_data,
    DataColumn<R> groups,
    R             group_1,
    R             group_2);
  template<typename T, typename R >
  std::tuple<R, R> sort_groups_by_threshold(
    Data<T>       data,
    DataColumn<R> groups,
    R             group_1,
    R             group_2,
    Projector<T>  projector,
    Threshold<T>  threshold);
  template<typename T, typename R >
  Condition<T, R> * binary_step(
    Data<T>          data,
    DataColumn<R>    groups,
    R                group_1,
    R                group_2,
    PPStrategy<T, R> pp_strategy);
  template<typename R >
  std::tuple<R, R> take_two(std::set<R> group_set);
  template<typename T, typename R >
  Node<T, R> * build_branch(
    Data<T>                     data,
    DataColumn<R>               groups,
    DataColumn<R>               binary_groups,
    R                           binary_group,
    std::map<int, std::set<R> > binary_group_mapping,
    PPStrategy<T, R>            pp_strategy);
  template<typename T, typename R >
  Condition<T, R> * step(
    Data<T>          data,
    DataColumn<R>    groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy);



  template<typename T, typename R >
  std::tuple<DataColumn<R>, std::set<int>, std::map<int, std::set<R> > >as_binary_problem(
    Data<T>          data,
    DataColumn<R>    groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy) {
    LOG_INFO << "Redefining a " << unique_groups.size() << " group problem as binary:" << std::endl;

    auto [projector, projected] = pp_strategy(data, groups, unique_groups);
    auto [binary_groups, binary_unique_groups, binary_group_mapping] = binary_regroup((Data<T>)projected, groups, unique_groups);

    LOG_INFO << "Mapping: " << binary_group_mapping << std::endl;
    return std::make_tuple(binary_groups, binary_unique_groups, binary_group_mapping);
  }

  template<typename T, typename R >
  Threshold<T> get_threshold(
    DataColumn<T> projected_data,
    DataColumn<R> groups,
    R             group_1,
    R             group_2) {
    T mean_1 = linalg::mean(select_group((Data<T>)projected_data, groups, group_1)).value();
    T mean_2 = linalg::mean(select_group((Data<T>)projected_data, groups, group_2)).value();

    return (mean_1 + mean_2) / 2;
  };

  template<typename T, typename R >
  std::tuple<R, R> sort_groups_by_threshold(
    Data<T>       data,
    DataColumn<R> groups,
    R             group_1,
    R             group_2,
    Projector<T>  projector,
    Threshold<T>  threshold) {
    R l_group, r_group;

    DataColumn<T> mean_1 = linalg::mean(select_group(data, groups, group_1));
    DataColumn<T> mean_2 = linalg::mean(select_group(data, groups, group_2));

    T projected_mean_1 = project(mean_1, projector);
    T projected_mean_2 = project(mean_2, projector);

    if (std::max(projected_mean_1, projected_mean_2) < threshold) {
      throw std::invalid_argument("Threshold is greater than the two groups means");
    }

    if (std::min(projected_mean_1, projected_mean_2) > threshold) {
      throw std::invalid_argument("Threshold is lower than the two groups means");
    }

    if (projected_mean_1 < projected_mean_2) {
      l_group = group_1;
      r_group = group_2;
    } else {
      l_group = group_2;
      r_group = group_1;
    }

    return std::make_tuple(group_1, group_2);
  }

  template<typename T, typename R >
  Condition<T, R> * binary_step(
    Data<T>          data,
    DataColumn<R>    groups,
    R                group_1,
    R                group_2,
    PPStrategy<T, R> pp_strategy) {
    std::set<R> unique_groups = { group_1, group_2 };
    LOG_INFO << "Project-Pursuit Tree building binary step for groups: " << unique_groups << std::endl;

    auto [projector, projected] = pp_strategy(data, groups, unique_groups);

    T threshold = get_threshold(projected, groups, group_1, group_2);

    auto [lower_group, upper_group] = sort_groups_by_threshold(
      data,
      groups,
      group_1,
      group_2,
      projector,
      threshold);

    Response<T, R> *lower_response = new Response<T, R>(lower_group);
    Response<T, R> *upper_response = new Response<T, R>(upper_group);

    Condition<T, R> *condition = new Condition<T, R>(
      projector,
      threshold,
      lower_response,
      upper_response);

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  }

  template<typename R >
  std::tuple<R, R> take_two(std::set<R> group_set) {
    if (group_set.size() < 2) {
      throw std::runtime_error("The set does not contain enough elements.");
    }

    auto first = *group_set.begin();
    auto last = *std::prev(group_set.end());
    return std::make_tuple(first, last);
  }

  template<typename T, typename R >
  Node<T, R> * build_branch(
    Data<T>                     data,
    DataColumn<R>               groups,
    DataColumn<R>               binary_groups,
    R                           binary_group,
    std::map<int, std::set<R> > binary_group_mapping,
    PPStrategy<T, R>            pp_strategy) {
    Node<T, R> *branch;
    std::set<R> unique_groups = binary_group_mapping[binary_group];

    if (unique_groups.size() == 1) {
      R group = *unique_groups.begin();
      LOG_INFO << "Branch is a Response for group " << group << std::endl;
      branch = new Response<T, R>(group);
    } else {
      LOG_INFO << "Branch is a Condition for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
      branch = step(
        select_group(data, binary_groups, binary_group),
        (DataColumn<R>)select_group((Data<R>)groups, binary_groups, binary_group),
        unique_groups,
        pp_strategy);
    }

    return branch;
  }

  template<typename T, typename R >
  Condition<T, R> * step(
    Data<T>          data,
    DataColumn<R>    groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy) {
    LOG_INFO << "Project-Pursuit Tree building step for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    if (unique_groups.size() == 2) {
      auto [group_1, group_2] = take_two(unique_groups);

      return binary_step(
        data,
        groups,
        group_1,
        group_2,
        pp_strategy);
    }

    auto [binary_groups, binary_unique_groups, binary_group_mapping] = as_binary_problem(
      data,
      groups,
      unique_groups,
      pp_strategy);

    auto [group_1, group_2] = take_two(binary_unique_groups);

    Condition<T, R> *temp_node = binary_step(
      data,
      binary_groups,
      group_1,
      group_2,
      pp_strategy);

    R binary_lower_group = as_response(temp_node->lower)->value;
    R binary_upper_group = as_response(temp_node->upper)->value;

    LOG_INFO << "Build lower branch" << std::endl;
    Node<T, R> *lower_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_lower_group,
      binary_group_mapping,
      pp_strategy);

    LOG_INFO << "Build upper branch" << std::endl;
    Node<T, R> *upper_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_upper_group,
      binary_group_mapping,
      pp_strategy);

    Condition<T, R> *condition = new Condition<T, R>(
      temp_node->projector,
      temp_node->threshold,
      lower_branch,
      upper_branch);

    delete temp_node;

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  };

  template<typename T, typename R >
  Tree<T, R> train(
    Data<T>          data,
    DataColumn<R>    groups,
    PPStrategy<T, R> pp_strategy) {
    std::set<R> unique_groups = unique(groups);

    LOG_INFO << "Project-Pursuit Tree training." << std::endl;
    Tree<T, R> tree = Tree(step(data, groups, unique_groups, pp_strategy));
    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<double, int> train(
    Data<double>            data,
    DataColumn<int>         groups,
    PPStrategy<double, int> pp_strategy);
}
