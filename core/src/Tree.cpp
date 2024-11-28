#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"
#include "BootstrapTree.hpp"
#include "Group.hpp"
#include "Logger.hpp"
#include "Invariant.hpp"

using namespace models::pp;
using namespace models::pp::strategy;
using namespace models::dr::strategy;
using namespace models::stats;
namespace models {
  template<typename T, typename R >
  std::unique_ptr<Condition<T, R> > step(
    const TrainingSpec<T, R> &  training_spec,
    const SortedDataSpec<T, R> &training_data);


  template<typename R >
  std::tuple<R, R> take_two(const std::set<R> &group_set) {
    invariant(group_set.size() >= 2, "The set does not contain enough elements.");

    auto first = *group_set.begin();
    auto last = *std::prev(group_set.end());
    return { first, last };
  }

  template<typename T, typename R >
  std::unique_ptr<Condition<T, R> > binary_step(
    const TrainingSpec<T, R> &   training_spec,
    const SortedDataSpec<T, R> & training_data) {
    auto [data, groups, unique_groups] = training_data.unwrap();
    auto [group_1, group_2] = take_two(unique_groups);

    LOG_INFO << "Project-Pursuit Tree building binary step for groups: " << unique_groups << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

    auto [projector, _] = pp_strategy(data, groups, unique_groups);

    Data<T> data_group_1 = training_data.group(group_1);
    Data<T> data_group_2 = training_data.group(group_2);

    T mean_1 = mean(project(data_group_1, projector));
    T mean_2 = mean(project(data_group_2, projector));

    LOG_INFO << "Mean for projected group " << group_1 << ": " << mean_1 << std::endl;
    LOG_INFO << "Mean for projected group " << group_2 << ": " << mean_2 << std::endl;

    T threshold =  (mean_1 + mean_2) / 2;

    LOG_INFO << "Threshold: " << threshold << std::endl;

    T projected_mean_1 = project(mean(data_group_1), projector);
    T projected_mean_2 = project(mean(data_group_2), projector);

    LOG_INFO << "Projected mean for group " << group_1 << ": " << projected_mean_1 << std::endl;
    LOG_INFO << "Projected mean for group " << group_2 << ": " << projected_mean_2 << std::endl;

    invariant(std::max(projected_mean_1, projected_mean_2) > threshold, "Threshold is greater than the two groups means");
    invariant(std::min(projected_mean_1, projected_mean_2) < threshold, "Threshold is lower than the two groups means");

    R lower_group, upper_group;

    if (projected_mean_1 < projected_mean_2) {
      lower_group = group_1;
      upper_group = group_2;
    } else {
      lower_group = group_2;
      upper_group = group_1;
    }

    LOG_INFO << "Lower group: " << lower_group << std::endl;
    LOG_INFO << "Upper group: " << upper_group << std::endl;

    std::unique_ptr<Node<T, R> >lower_response = std::make_unique<Response<T, R> >(lower_group);
    std::unique_ptr<Node<T, R> >upper_response = std::make_unique<Response<T, R> >(upper_group);

    std::unique_ptr<Condition<T, R> > condition = std::make_unique<Condition<T, R> >(
      projector,
      threshold,
      std::move(lower_response),
      std::move(upper_response),
      training_spec.clone(),
      std::make_unique<SortedDataSpec<T, R> >(data, groups, unique_groups));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
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

    SortedDataSpec<T, R> training_data(
      select_group(data, binary_groups, binary_group),
      select_group(groups, binary_groups, binary_group),
      unique_groups);

    return step(training_spec, training_data);
  }

  template<typename T, typename R >
  std::unique_ptr< Condition<T, R> >   step(
    const TrainingSpec<T, R> &  training_spec,
    const SortedDataSpec<T, R> &training_data) {
    auto data = training_data.x;
    auto groups = training_data.y;
    auto unique_groups = training_data.classes;

    LOG_INFO << "Project-Pursuit Tree building step for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T> &dr_strategy = *(training_spec.dr_strategy);

    Data<T> reduced_data = dr_strategy(data);

    if (unique_groups.size() == 2) {
      return binary_step(
        training_spec,
        SortedDataSpec<T, R>(
          reduced_data,
          groups,
          unique_groups));
    }

    LOG_INFO << "Redefining a " << unique_groups.size() << " group problem as binary:" << std::endl;

    auto [projector, projected] = pp_strategy(data, groups, unique_groups);
    auto [binary_groups, binary_unique_groups, binary_group_mapping] = binary_regroup((Data<T>)projected, groups, unique_groups);

    LOG_INFO << "Mapping: " << binary_group_mapping << std::endl;

    std::unique_ptr<Condition<T, R> > temp_node = binary_step(
      training_spec,
      SortedDataSpec<T, R>(
        reduced_data,
        groups,
        binary_unique_groups));

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
      std::make_unique<SortedDataSpec<T, R> >(data, groups, unique_groups));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  };

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  DerivedTree<T, R> BaseTree<T, R, D, DerivedTree>::train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data) {
    LOG_INFO << "Project-Pursuit Tree training." << std::endl;

    auto unwrapped = training_data.unwrap();
    auto &x = std::get<0>(unwrapped);
    auto &y = std::get<1>(unwrapped);
    auto &classes = std::get<2>(unwrapped);

    LOG_INFO << "Root step." << std::endl;
    auto root_ptr = step(training_spec, training_data);


    DerivedTree<T, R> tree(
      std::move(root_ptr),
      training_spec.clone(),
      std::make_shared<D >(training_data));

    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<double, int> BaseTree<double, int, SortedDataSpec<double, int>, Tree>::train(
    const TrainingSpec<double, int> &   training_spec,
    const SortedDataSpec<double, int> & training_data);

  template BootstrapTree<double, int> BaseTree<double, int, BootstrapDataSpec<double, int>, BootstrapTree>::train(
    const TrainingSpec<double, int> &     training_spec,
    const BootstrapDataSpec<double, int> &training_data);
}
