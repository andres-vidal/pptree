#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"
#include "BootstrapTree.hpp"
#include "Logger.hpp"
#include "Invariant.hpp"
#include "Map.hpp"

using namespace models::pp;
using namespace models::pp::strategy;
using namespace models::dr::strategy;
using namespace models::stats;
using namespace utils;

namespace models {
  template<typename T, typename R >
  std::map<R, int> binary_regroup(
    const SortedDataSpec<T, R> data
    ) {
    std::vector<std::tuple<R, T> > means;

    for (const R group : data.classes) {
      Data<T> group_data = data.group(group);
      means.push_back({ group, mean(group_data).value() });
    }

    std::sort(means.begin(), means.end(), [](const auto &a, const auto &b) {
       return std::get<1>(a) < std::get<1>(b);
     });

    T edge_gap = 0;
    T edge_group = 0;

    for (int i = 0; i < means.size() - 1; i++) {
      T gap = std::get<1>(means[i + 1]) - std::get<1>(means[i]);
      LOG_INFO << "Gap between " << std::get<0>(means[i]) << " and " << std::get<0>(means[i + 1]) << ": " << gap << std::endl;

      if (gap > edge_gap) {
        edge_gap = gap;
        edge_group = std::get<0>(means[i + 1]);

        LOG_INFO << "New edge gap: " << edge_gap << std::endl;
        LOG_INFO << "New edge group: " << edge_group << std::endl;
      }
    }

    LOG_INFO << "Edge group: " << edge_group << std::endl;

    std::map<R, int > binary_mapping;

    bool edge_found = false;
    for (const auto&[group, mean] : means) {
      LOG_INFO << "Remapping group " << group << std::endl;

      edge_found = edge_found || group == edge_group;

      binary_mapping[group] = edge_found ? 1 : 0;

      LOG_INFO << "Mapping: " << group << " -> " << binary_mapping[group] << std::endl;
    }

    return binary_mapping;
  }

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

  template<typename T, typename R>
  std::unique_ptr<Node<T, R> >   step(
    const TrainingSpec<T, R> &     training_spec,
    const BootstrapDataSpec<T, R> &training_data) {
    return step(training_spec, training_data.get_sample());
  }

  template<typename T, typename R >
  std::unique_ptr<Node<T, R> >   step(
    const TrainingSpec<T, R> &  training_spec,
    const SortedDataSpec<T, R> &training_data) {
    auto[data, groups, unique_groups] = training_data.unwrap();

    LOG_INFO << "Project-Pursuit Tree building step for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T> &dr_strategy = *(training_spec.dr_strategy);

    if (unique_groups.size() == 1) {
      return std::make_unique<Response<T, R> >(*unique_groups.begin());
    }

    SortedDataSpec<T, R> reduced_data = training_data.analog(dr_strategy(data));

    if (unique_groups.size() == 2) {
      return binary_step(training_spec, reduced_data);
    }

    LOG_INFO << "Redefining a " << unique_groups.size() << " group problem as binary:" << std::endl;

    auto [projector, projected] = pp_strategy(reduced_data.x, groups, unique_groups);

    SortedDataSpec<T, R> projected_reduced_data = reduced_data.analog(projected);

    std::map<R, int> binary_mapping = binary_regroup(projected_reduced_data);

    LOG_INFO << "Mapping: " << binary_mapping << std::endl;

    SortedDataSpec<T, R> binary_remapped_data = reduced_data.remap(binary_mapping);

    std::unique_ptr<Condition<T, R> > temp_node = binary_step(
      training_spec,
      binary_remapped_data);

    R binary_lower_group = temp_node->lower->response();
    R binary_upper_group = temp_node->upper->response();

    std::map<int, std::set<R> > inverse_mapping = invert(binary_mapping);
    std::set<R> lower_groups = inverse_mapping.at(binary_lower_group);
    std::set<R> upper_groups = inverse_mapping.at(binary_upper_group);

    LOG_INFO << "Build lower branch" << std::endl;
    std::unique_ptr<Node<T, R> > lower_branch = step(training_spec, training_data.subset(lower_groups));

    LOG_INFO << "Build upper branch" << std::endl;
    std::unique_ptr<Node<T, R> > upper_branch = step(training_spec, training_data.subset(upper_groups));

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
