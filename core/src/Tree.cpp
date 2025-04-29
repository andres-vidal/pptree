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
    const SortedDataSpec<T, R> &data
    ) {
    std::vector<std::tuple<R, T> > means;

    invariant(data.x.cols() == 1, "Binary regrouping requires a unidimensional data");

    for (const R group : data.classes) {
      Data<T> group_data = data.group(group);

      T group_mean = group_data.mean();

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
  std::unique_ptr<Condition<T, R> > binary_step(
    const TrainingSpec<T, R> &   training_spec,
    const SortedDataSpec<T, R> & training_data,
    const ReducedDataSpec<T, R>& reduced_data) {
    R group_1 = *training_data.classes.begin();
    R group_2 = *std::next(training_data.classes.begin());

    LOG_INFO << "Project-Pursuit Tree building binary step for groups: " << training_data.classes << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);

    auto projector = reduced_data.expand(pp_strategy(reduced_data));

    Data<T> data_group_1 = training_data.group(group_1);
    Data<T> data_group_2 = training_data.group(group_2);

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

    auto lower_response = std::make_unique<Response<T, R> >(lower_group);
    auto upper_response = std::make_unique<Response<T, R> >(upper_group);

    auto condition = std::make_unique<Condition<T, R> >(
      projector,
      threshold,
      std::move(lower_response),
      std::move(upper_response),
      training_spec.clone(),
      std::make_unique<SortedDataSpec<T, R> >(training_data));

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
    LOG_INFO << "Project-Pursuit Tree building step for " << training_data.classes.size() << " groups: " << training_data.classes << std::endl;
    LOG_INFO << "Dataset size: " << training_data.x.rows() << " observations of " << training_data.x.cols() << " variables" << std::endl;

    const PPStrategy<T, R> &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy<T, R> &dr_strategy = *(training_spec.dr_strategy);

    if (training_data.classes.size() == 1) {
      return std::make_unique<Response<T, R> >(*training_data.classes.begin());
    }

    auto reduced_data = dr_strategy(training_data);

    if (training_data.classes.size() == 2) {
      return binary_step(training_spec, training_data, reduced_data);
    }

    LOG_INFO << "Redefining a " << training_data.classes.size() << " group problem as binary:" << std::endl;

    auto projector      = reduced_data.expand(pp_strategy(reduced_data));
    auto binary_mapping = binary_regroup(training_data.analog(project(training_data.x, projector)));

    LOG_INFO << "Mapping: " << binary_mapping << std::endl;

    auto binary_training_data = training_data.remap(binary_mapping);
    auto binary_reduced_data  = reduced_data.remap(binary_mapping);
    auto temp_node            = binary_step(training_spec, binary_training_data, binary_reduced_data);

    R binary_lower_group = temp_node->lower->response();
    R binary_upper_group = temp_node->upper->response();

    std::map<R, std::set<R> > inverse_mapping = invert(binary_mapping);
    auto lower_groups                         = inverse_mapping.at(binary_lower_group);
    auto upper_groups                         = inverse_mapping.at(binary_upper_group);

    LOG_INFO << "Build lower branch" << std::endl;
    auto lower_branch = step(training_spec, training_data.subset(lower_groups));

    LOG_INFO << "Build upper branch" << std::endl;
    auto upper_branch = step(training_spec, training_data.subset(upper_groups));

    auto condition = std::make_unique<Condition<T, R> >(
      temp_node->projector,
      temp_node->threshold,
      std::move(lower_branch),
      std::move(upper_branch),
      training_spec.clone(),
      std::make_unique<SortedDataSpec<T, R> >(training_data));

    LOG_INFO << "Condition: " << *condition << std::endl;
    return condition;
  };

  template<typename T, typename R, typename D, template<typename, typename> class DerivedTree>
  DerivedTree<T, R> BaseTree<T, R, D, DerivedTree>::train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data) {
    LOG_INFO << "Project-Pursuit Tree training." << std::endl;

    LOG_INFO << "Root step." << std::endl;
    auto root_ptr = step(training_spec, training_data);

    DerivedTree<T, R> tree(
      std::move(root_ptr),
      training_spec.clone(),
      std::make_shared<D >(training_data));

    LOG_INFO << "Tree: " << tree << std::endl;
    return tree;
  }

  template Tree<float, int> BaseTree<float, int, SortedDataSpec<float, int>, Tree>::train(
    const TrainingSpec<float, int> &   training_spec,
    const SortedDataSpec<float, int> & training_data);

  template BootstrapTree<float, int> BaseTree<float, int, BootstrapDataSpec<float, int>, BootstrapTree>::train(
    const TrainingSpec<float, int> &     training_spec,
    const BootstrapDataSpec<float, int> &training_data);
}
