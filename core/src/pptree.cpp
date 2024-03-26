#include "pptree.hpp"
#include "pptreeio.hpp"
#include <vector>
#include <set>

using namespace pptree;
using namespace Eigen;

namespace pptree {
  template<typename T >
  using DimensionalityReductionStrategy = std::function<const Data<T> & (const Data<T>&)>;

  template<typename T >
  const Data<T> & select_all_variables(const Data<T> &data) {
    return data;
  }

  template<typename T, typename R >
  std::unique_ptr<Condition<T, R> > step(
    const Data<T> &                           data,
    const DataColumn<R> &                     groups,
    const std::set<R> &                       unique_groups,
    const PPStrategy<T, R> &                  pp_strategy,
    const DimensionalityReductionStrategy<T> &reduce_dimensions);

  template<typename T, typename R >
  std::tuple<DataColumn<R>, std::set<int>, std::map<int, std::set<R> > >as_binary_problem(
    const Data<T> &         data,
    const DataColumn<R> &   groups,
    const std::set<R> &     unique_groups,
    const PPStrategy<T, R> &pp_strategy) {
    LOG_INFO << "Redefining a " << unique_groups.size() << " group problem as binary:" << std::endl;

    auto [projector, projected] = pp_strategy(data, groups, unique_groups);
    auto [binary_groups, binary_unique_groups, binary_group_mapping] = binary_regroup((Data<T>)projected, groups, unique_groups);

    LOG_INFO << "Mapping: " << binary_group_mapping << std::endl;
    return { binary_groups, binary_unique_groups, binary_group_mapping };
  }

  template<typename T, typename R >
  Threshold<T> get_threshold(
    const DataColumn<T> &projected_data,
    const DataColumn<R> &groups,
    const R &            group_1,
    const R &            group_2) {
    T mean_1 = linalg::mean(select_group(projected_data, groups, group_1)).value();
    T mean_2 = linalg::mean(select_group(projected_data, groups, group_2)).value();

    return (mean_1 + mean_2) / 2;
  };

  template<typename T, typename R >
  std::tuple<R, R> sort_groups_by_threshold(
    const Data<T> &      data,
    const DataColumn<R> &groups,
    const R &            group_1,
    const R &            group_2,
    const Projector<T> & projector,
    const Threshold<T> & threshold) {
    LOG_INFO << "Sorting groups by threshold:" << std::endl;
    LOG_INFO << "Threshold: " << threshold << std::endl;

    R l_group, u_group;

    DataColumn<T> mean_1 = linalg::mean(select_group(data, groups, group_1));
    DataColumn<T> mean_2 = linalg::mean(select_group(data, groups, group_2));

    T projected_mean_1 = project(mean_1, projector);
    T projected_mean_2 = project(mean_2, projector);

    LOG_INFO << "Projected mean for group " << group_1 << ": " << projected_mean_1 << std::endl;
    LOG_INFO << "Projected mean for group " << group_2 << ": " << projected_mean_2 << std::endl;

    assert(std::max(projected_mean_1, projected_mean_2) > threshold && "Threshold is greater than the two groups means");
    assert(std::min(projected_mean_1, projected_mean_2) < threshold && "Threshold is lower than the two groups means");

    if (projected_mean_1 < projected_mean_2) {
      l_group = group_1;
      u_group = group_2;
    } else {
      l_group = group_2;
      u_group = group_1;
    }

    LOG_INFO << "Lower group: " << l_group << std::endl;
    LOG_INFO << "Upper group: " << u_group << std::endl;

    return { l_group, u_group };
  }

  template<typename T, typename R >
  std::unique_ptr<Condition<T, R> > binary_step(
    const Data<T> &         data,
    const DataColumn<R> &   groups,
    const R &               group_1,
    const R &               group_2,
    const PPStrategy<T, R> &pp_strategy) {
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

    std::unique_ptr<Node<T, R> >lower_response = std::make_unique<Response<T, R> >(lower_group);
    std::unique_ptr<Node<T, R> >upper_response = std::make_unique<Response<T, R> >(upper_group);

    std::unique_ptr<Condition<T, R> > condition = std::make_unique<Condition<T, R> >(
      projector,
      threshold,
      std::move(lower_response),
      std::move(upper_response));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return std::move(condition);
  }

  template<typename R >
  std::tuple<R, R> take_two(const std::set<R> &group_set) {
    assert(group_set.size() >= 2 && "The set does not contain enough elements.");

    auto first = *group_set.begin();
    auto last = *std::prev(group_set.end());
    return { first, last };
  }

  template<typename T, typename R >
  std::unique_ptr<Node<T, R> > build_branch(
    const Data<T> &                           data,
    const DataColumn<R> &                     groups,
    const DataColumn<R> &                     binary_groups,
    const R &                                 binary_group,
    const std::map<int, std::set<R> >&        binary_group_mapping,
    const PPStrategy<T, R> &                  pp_strategy,
    const DimensionalityReductionStrategy<T> &reduce_dimensions) {
    std::set<R> unique_groups = binary_group_mapping.at(binary_group);

    if (unique_groups.size() == 1) {
      R group = *unique_groups.begin();
      LOG_INFO << "Branch is a Response for group " << group << std::endl;
      return std::move(std::make_unique<Response<T, R> >(group));
    }

    LOG_INFO << "Branch is a Condition for " << unique_groups.size() << " groups: " << unique_groups << std::endl;

    std::unique_ptr<Condition<T, R> > condition = step(
      select_group(data, binary_groups, binary_group),
      select_group(groups, binary_groups, binary_group),
      unique_groups,
      pp_strategy,
      reduce_dimensions);

    return std::move(condition);
  }

  template<typename T, typename R >
  std::unique_ptr< Condition<T, R> >  step(
    const Data<T> &                            data,
    const DataColumn<R> &                      groups,
    const std::set<R> &                        unique_groups,
    const PPStrategy<T, R> &                   pp_strategy,
    const DimensionalityReductionStrategy<T> & reduce_dimensions) {
    LOG_INFO << "Project-Pursuit Tree building step for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    Data<T> reduced_data = reduce_dimensions(data);

    if (unique_groups.size() == 2) {
      auto [group_1, group_2] = take_two(unique_groups);

      return binary_step(
        reduced_data,
        groups,
        group_1,
        group_2,
        pp_strategy);
    }

    auto [binary_groups, binary_unique_groups, binary_group_mapping] = as_binary_problem(
      reduced_data,
      groups,
      unique_groups,
      pp_strategy);

    auto [group_1, group_2] = take_two(binary_unique_groups);

    std::unique_ptr<Condition<T, R> > temp_node = binary_step(
      reduced_data,
      binary_groups,
      group_1,
      group_2,
      pp_strategy);

    R binary_lower_group = temp_node->lower->as_response().value;
    R binary_upper_group = temp_node->upper->as_response().value;

    LOG_INFO << "Build lower branch" << std::endl;
    std::unique_ptr<Node<T, R> > lower_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_lower_group,
      binary_group_mapping,
      pp_strategy,
      reduce_dimensions);

    LOG_INFO << "Build upper branch" << std::endl;
    std::unique_ptr<Node<T, R> > upper_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_upper_group,
      binary_group_mapping,
      pp_strategy,
      reduce_dimensions);

    std::unique_ptr<Condition<T, R> > condition = std::make_unique<Condition<T, R> >(
      temp_node->projector,
      temp_node->threshold,
      std::move(lower_branch),
      std::move(upper_branch));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return std::move(condition);
  };

  template<typename T, typename R >
  Tree<T, R> train(
    const Data<T> &         data,
    const DataColumn<R> &   groups,
    const PPStrategy<T, R> &pp_strategy) {
    std::set<R> unique_groups = unique(groups);

    LOG_INFO << "Project-Pursuit Tree training." << std::endl;
    Tree<T, R> tree = Tree(step(data, groups, unique_groups, pp_strategy, (DimensionalityReductionStrategy<T>)select_all_variables<T>));
    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template<typename T, typename R>
  Tree<T, R> train_glda(
    const Data<T> &      data,
    const DataColumn<R> &groups,
    const double         lambda) {
    return train(data, groups, (PPStrategy<T, R>)glda_strategy<T, R>(lambda));
  }

  template Tree<long double, int> train_glda(
    const Data<long double> &data,
    const DataColumn<int> &  groups,
    const double             lambda);
}
