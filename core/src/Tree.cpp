#include "Tree.hpp"

#include "BootstrapTree.hpp"
#include "Logger.hpp"
#include "Invariant.hpp"
#include "Map.hpp"

#include <stack>

using namespace models::pp;
using namespace models::pp::strategy;
using namespace models::dr::strategy;
using namespace models::stats;
using namespace utils;

namespace models {
  template<typename T, typename R >
  std::map<R, int> binary_regroup(
    const Data<T> &     x,
    const GroupSpec<R> &data_spec
    ) {
    std::vector<std::tuple<R, T> > means;

    invariant(x.cols() == 1, "Binary regrouping requires a unidimensional data");

    for (const R group : data_spec.groups) {
      T group_mean = data_spec.group(x, group).mean();

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
    const Data<T> &            x,
    const GroupSpec<R> &       data_spec,
    const DRSpec<T, R>&        dr) {
    R group_1 = *data_spec.groups.begin();
    R group_2 = *std::next(data_spec.groups.begin());

    LOG_INFO << "Project-Pursuit Tree building binary recursive_step for groups: " << data_spec.groups << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

    auto reduced_x = x(Eigen::all, dr.selected_cols);

    Projector<T> projector = dr.expand(pp_strategy(reduced_x, data_spec));

    auto data_group_1 = data_spec.group(x, group_1);
    auto data_group_2 = data_spec.group(x, group_2);

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
      data_spec.groups);

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  }

  template<typename T, typename R >
  GroupSpec<R> binary_split(
    const TrainingSpec<T, R> & training_spec,
    const Data<T> &            x,
    const GroupSpec<R> &       data_spec,
    const DRSpec<T, R>&        dr
    ) {
    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

    LOG_INFO << "Redefining a " << data_spec.groups.size() << " group problem as binary:" << std::endl;

    Data<T> reduced_x = x(Eigen::all, dr.selected_cols);

    Projector<T> projector = dr.expand(pp_strategy(reduced_x, data_spec));

    Data<T> projected_x = x * projector;

    std::map<R, int> binary_mapping = binary_regroup(projected_x, data_spec);

    LOG_INFO << "Mapping: " << binary_mapping << std::endl;

    return data_spec.remap(binary_mapping);
  }

  template<typename T, typename R>
  struct Step {
    GroupSpec<R> y;
    TreeNodePtr<T, R> *node;

    bool pop                = false;
    TreeNodePtr<T, R> upper = nullptr;
    TreeNodePtr<T, R> lower = nullptr;
    Threshold<T> threshold  = 0;
    Projector<T> projector;

    Step(
      const GroupSpec<R>& y,
      TreeNodePtr<T, R>   *node,
      const int           cols
      ) :
      y(y),
      node(node),
      projector(Projector<T>::Zero(cols)) {
    }
  };

  template<typename T, typename R>
  TreeNodePtr<T, R> build_root(
    const TrainingSpec<T, R> & training_spec,
    const Data<T> &            x,
    const GroupSpec<R> &       y
    ) {
    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T, R> &dr_strategy = *(training_spec.dr_strategy);

    std::stack<Step<T, R> > stack;

    TreeNodePtr<T, R> root;

    stack.emplace(y, &root, x.cols());

    while (!stack.empty()) {
      Step<T, R>& step = stack.top();

      if (step.pop) {
        *step.node = TreeCondition<T, R>::make(
          step.projector,
          step.threshold,
          std::move(step.lower),
          std::move(step.upper),
          training_spec.clone(),
          step.y.groups);

        stack.pop();
        continue;
      }

      if (step.y.groups.size() == 1) {
        *step.node = TreeResponse<T, R>::make(*step.y.groups.begin());
        stack.pop();
        continue;
      }

      DRSpec<T, R> dr = dr_strategy(x, step.y);

      if (step.y.groups.size() == 2) {
        *step.node = binary_step(training_spec, x, step.y, dr);
        stack.pop();
        continue;
      }

      auto split     = binary_split(training_spec, x, step.y, dr);
      auto temp_node = binary_step(training_spec, x, split, dr);

      R binary_lower_group = temp_node->lower->response();
      R binary_upper_group = temp_node->upper->response();

      step.projector = temp_node->projector;
      step.threshold = temp_node->threshold;

      auto lower_y = split.subset(split.subgroups.at(binary_lower_group));
      auto upper_y = split.subset(split.subgroups.at(binary_upper_group));

      stack.emplace(lower_y, &step.lower, x.cols());
      stack.emplace(upper_y, &step.upper, x.cols());

      step.pop = true;
    }

    return root;
  }

  template<typename T, typename R>
  Tree<T, R> Tree<T, R>::train(
    const TrainingSpec<T, R> & training_spec,
    Data<T>&                   x,
    DataColumn<R>&             y) {
    LOG_INFO << "Project-Pursuit Tree training." << std::endl;

    if (!GroupSpec<R>::is_contiguous(y)) {
      models::stats::sort(x, y);
    }

    GroupSpec<R> data_spec(y);

    LOG_INFO << "Root recursive_step." << std::endl;
    TreeNodePtr<T, R> root_ptr = build_root(training_spec, x, data_spec);

    Tree<T, R> tree(
      std::move(root_ptr),
      training_spec.clone(),
      x,
      y,
      data_spec.groups);

    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<float, int> Tree<float, int>::train(
    const TrainingSpec<float, int> & training_spec,
    Data<float>&                     x,
    DataColumn<int>&                 y);
}
