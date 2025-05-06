#include "Tree.hpp"

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
    const DataSpec<T, R> &training_data
    ) {
    std::vector<std::tuple<R, T> > means;

    invariant(training_data.cols() == 1, "Binary regrouping requires a unidimensional data");

    for (const R group : training_data.groups) {
      T group_mean = training_data.group(group).mean();

      means.push_back({ group, group_mean });
    }

    std::sort(means.begin(), means.end(), [](const auto &a, const auto &b) {
        return std::get<1>(a) < std::get<1>(b);
      });

    T edge_gap   = -1;
    R edge_group = -1;

    for (int i = 0; i < means.size() - 1; i++) {
      T gap = std::get<1>(means[i + 1]) - std::get<1>(means[i]);
      LOG_INFO << "Gap between " << std::get<0>(means[i]) << " and " << std::get<0>(means[i + 1]) << ": " << gap << std::endl;

      if (gap > edge_gap) {
        edge_gap   = gap;
        edge_group = std::get<0>(means[i + 1]);

        LOG_INFO << "New edge gap: " << edge_gap << std::endl;
        LOG_INFO << "New edge group: " << edge_group << std::endl;
      }
    }

    if (edge_group == -1) {
      LOG_INFO << "Edge group not found. Using first group." << std::endl;
      edge_group = std::get<0>(means.front());
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

  template<typename T, typename R >
  TreeConditionPtr<T, R> binary_step(
    const TrainingSpec<T, R> & training_spec,
    const DataSpec<T, R> &     training_data,
    const DRSpec<T, R>&        dr) {
    R group_1 = *training_data.groups.begin();
    R group_2 = *std::next(training_data.groups.begin());

    LOG_INFO << "Project-Pursuit Tree building binary step for groups: " << training_data.groups << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

    Data<T> reduced_x = training_data.x(Eigen::all, dr.selected_cols);

    Projector<T> projector = dr.expand(pp_strategy(training_data.analog(reduced_x)));

    auto data_group_1 = training_data.group(group_1);
    auto data_group_2 = training_data.group(group_2);

    T mean_1 = (data_group_1 * projector).mean();
    T mean_2 = (data_group_2 * projector).mean();

    LOG_INFO << "Mean for projected group " << group_1 << ": " << mean_1 << std::endl;
    LOG_INFO << "Mean for projected group " << group_2 << ": " << mean_2 << std::endl;

    T threshold =  (mean_1 + mean_2) / 2;

    LOG_INFO << "Threshold: " << threshold << std::endl;

    T projected_mean_1 = data_group_1.colwise().mean().dot(projector);
    T projected_mean_2 = data_group_2.colwise().mean().dot(projector);

    LOG_INFO << "Projected mean for group " << group_1 << ": " << projected_mean_1 << std::endl;
    LOG_INFO << "Projected mean for group " << group_2 << ": " << projected_mean_2 << std::endl;

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

    TreeResponsePtr<T, R> lower_response = TreeResponse<T, R>::make(lower_group);
    TreeResponsePtr<T, R> upper_response = TreeResponse<T, R>::make(upper_group);

    auto condition = TreeCondition<T, R>::make(
      projector,
      threshold,
      std::move(lower_response),
      std::move(upper_response),
      training_spec.clone(),
      training_data.groups);

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  }

  template<typename T, typename R >
  TreeNodePtr<T, R>   step(
    const TrainingSpec<T, R> & training_spec,
    const DataSpec<T, R> &     training_data) {
    LOG_INFO << "Project-Pursuit Tree building step for " << training_data.groups.size() << " groups: " << training_data.groups << std::endl;
    LOG_INFO << "Dataset size: " << training_data.x.rows() << " observations of " << training_data.x.cols() << " variables" << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T, R> &dr_strategy = *(training_spec.dr_strategy);

    if (training_data.groups.size() == 1) {
      return TreeResponse<T, R>::make(*training_data.groups.begin());
    }

    DRSpec<T, R> dr = dr_strategy(training_data);

    if (training_data.groups.size() == 2) {
      return binary_step(training_spec, training_data, dr);
    }

    LOG_INFO << "Redefining a " << training_data.groups.size() << " group problem as binary:" << std::endl;

    Data<T> reduced_x = training_data.x(Eigen::all, dr.selected_cols);

    Projector<T> projector = dr.expand(pp_strategy(training_data.analog(reduced_x)));

    Data<T> projected_x = training_data.x * projector;

    std::map<R, int> binary_mapping = binary_regroup(training_data.analog(projected_x));

    LOG_INFO << "Mapping: " << binary_mapping << std::endl;

    TreeConditionPtr<T, R> temp_node = binary_step(training_spec, training_data.remap(binary_mapping), dr);

    R binary_lower_group = temp_node->lower->response();
    R binary_upper_group = temp_node->upper->response();

    std::map<R, std::set<R> > inverse_mapping = invert(binary_mapping);
    std::set<R> lower_groups                  = inverse_mapping.at(binary_lower_group);
    std::set<R> upper_groups                  = inverse_mapping.at(binary_upper_group);

    LOG_INFO << "Build lower branch" << std::endl;
    TreeNodePtr<T, R> lower_branch = step(training_spec, training_data.subset(lower_groups));

    LOG_INFO << "Build upper branch" << std::endl;
    TreeNodePtr<T, R> upper_branch = step(training_spec, training_data.subset(upper_groups));

    auto condition = TreeCondition<T, R>::make(
      temp_node->projector,
      temp_node->threshold,
      std::move(lower_branch),
      std::move(upper_branch),
      training_spec.clone(),
      training_data.groups);

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  };

  template<typename T, typename R>
  Tree<T, R> Tree<T, R>::train(
    const TrainingSpec<T, R> & training_spec,
    Data<T>&                   x,
    DataColumn<R>&             y) {
    LOG_INFO << "Project-Pursuit Tree training." << std::endl;

    if (!DataSpec<T, R>::is_contiguous(y)) {
      models::stats::sort(x, y);
    }

    DataSpec<T, R> training_data(x, y);

    LOG_INFO << "Root step." << std::endl;
    TreeNodePtr<T, R> root_ptr = step(training_spec, training_data);

    Tree<T, R> tree(
      std::move(root_ptr),
      training_spec.clone(),
      x,
      y,
      training_data.groups);

    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<float, int> Tree<float, int>::train(
    const TrainingSpec<float, int> & training_spec,
    Data<float>&                     x,
    DataColumn<int>&                 y);
}
