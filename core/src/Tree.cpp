
#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"
#include "BootstrapTree.hpp"
#include "Group.hpp"
#include "Logger.hpp"


using namespace models::pp;
using namespace models::pp::strategy;
using namespace models::dr::strategy;
using namespace models::stats;
namespace models {
  template<typename T, typename R >
  std::unique_ptr<Condition<T, R> > step(
    const Data<T> &           data,
    const DataColumn<R> &     groups,
    const std::set<R> &       unique_groups,
    const TrainingSpec<T, R> &training_spec);

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
    T mean_1 = mean(select_group(projected_data, groups, group_1));
    T mean_2 = mean(select_group(projected_data, groups, group_2));

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

    DataColumn<T> mean_1 = mean(select_group(data, groups, group_1));
    DataColumn<T> mean_2 = mean(select_group(data, groups, group_2));

    T projected_mean_1 = projector.project(mean_1);
    T projected_mean_2 = projector.project(mean_2);

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
    const Data<T> &            data,
    const DataColumn<R> &      groups,
    const R &                  group_1,
    const R &                  group_2,
    const TrainingSpec<T, R> & training_spec) {
    std::set<R> unique_groups = { group_1, group_2 };
    LOG_INFO << "Project-Pursuit Tree building binary step for groups: " << unique_groups << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

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
      std::move(upper_response),
      training_spec.clone(),
      std::make_unique<DataSpec<T, R> >(data, groups, unique_groups));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
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
    const Data<T> &                    data,
    const DataColumn<R> &              groups,
    const DataColumn<R> &              binary_groups,
    const R &                          binary_group,
    const std::map<int, std::set<R> >& binary_group_mapping,
    const TrainingSpec<T, R> &         training_spec) {
    std::set<R> unique_groups = binary_group_mapping.at(binary_group);

    if (unique_groups.size() == 1) {
      R group = *unique_groups.begin();
      LOG_INFO << "Branch is a Response for group " << group << std::endl;
      return std::make_unique<Response<T, R> >(group);
    }

    LOG_INFO << "Branch is a Condition for " << unique_groups.size() << " groups: " << unique_groups << std::endl;

    std::unique_ptr<Condition<T, R> > condition = step(
      select_group(data, binary_groups, binary_group),
      select_group(groups, binary_groups, binary_group),
      unique_groups,
      training_spec);

    return condition;
  }

  template<typename T, typename R >
  std::unique_ptr< Condition<T, R> >  step(
    const Data<T> &            data,
    const DataColumn<R> &      groups,
    const std::set<R> &        unique_groups,
    const TrainingSpec<T, R> & training_spec) {
    LOG_INFO << "Project-Pursuit Tree building step for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T> &dr_strategy = *(training_spec.dr_strategy);

    Data<T> reduced_data = dr_strategy(data);

    if (unique_groups.size() == 2) {
      auto [group_1, group_2] = take_two(unique_groups);

      return binary_step(
        reduced_data,
        groups,
        group_1,
        group_2,
        training_spec);
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
      training_spec);

    R binary_lower_group = temp_node->lower->response();
    R binary_upper_group = temp_node->upper->response();

    LOG_INFO << "Build lower branch" << std::endl;
    std::unique_ptr<Node<T, R> > lower_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_lower_group,
      binary_group_mapping,
      training_spec);

    LOG_INFO << "Build upper branch" << std::endl;
    std::unique_ptr<Node<T, R> > upper_branch = build_branch(
      data,
      groups,
      binary_groups,
      binary_upper_group,
      binary_group_mapping,
      training_spec);

    std::unique_ptr<Condition<T, R> > condition = std::make_unique<Condition<T, R> >(
      temp_node->projector,
      temp_node->threshold,
      std::move(lower_branch),
      std::move(upper_branch),
      training_spec.clone(),
      std::make_unique<DataSpec<T, R> >(data, groups, unique_groups));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  };

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  DerivedTree<T, R> BaseTree<T, R, D, DerivedTree>::train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data) {
    LOG_INFO << "Project-Pursuit Tree training." << std::endl;

    auto [x, y, classes] = training_data.unwrap();

    DerivedTree<T, R> tree(
      step(
        x,
        y,
        classes,
        training_spec),
      training_spec.clone(),
      std::make_shared<D >(training_data));

    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<long double, int> BaseTree<long double, int, DataSpec<long double, int>, Tree>::train(
    const TrainingSpec<long double, int> &training_spec,
    const DataSpec<long double, int> &    training_data);

  template BootstrapTree<long double, int> BaseTree<long double, int, BootstrapDataSpec<long double, int>, BootstrapTree>::train(
    const TrainingSpec<long double, int> &     training_spec,
    const BootstrapDataSpec<long double, int> &training_data);
}
