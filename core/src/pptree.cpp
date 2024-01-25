#include "pptree.hpp"
#include <vector>
#include <set>

using namespace pp;
using namespace stats;
using namespace Eigen;

namespace pptree {
  template<typename T, typename R >
  std::tuple<DataColumn<R>, std::set<int>, std::map<int, std::set<R> > >as_binary_problem(
    Data<T>          data,
    DataColumn<R>    groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy) {
    auto [projector, projected] = pp_strategy(data, groups, unique_groups);
    return binary_regroup((Data<T>)projected, groups, unique_groups);
  }

  template std::tuple<DataColumn<int>, std::set<int>, std::map<int, std::set<int> > >as_binary_problem(
    Data<double>            data,
    DataColumn<int>         groups,
    std::set<int>           unique_groups,
    PPStrategy<double, int> pp_strategy);

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

  template Threshold<double> get_threshold(
    DataColumn<double> projected_data,
    DataColumn<int>    groups,
    int                group_1,
    int                group_2);

  template<typename T, typename R >
  std::tuple<R, R> sort_groups_by_threshold(
    Data<T>       data,
    DataColumn<R> groups,
    R             group_1,
    R             group_2,
    Projector<T>  projector,
    Threshold<T>  threshold) {
    R l_group, r_group;

    Data<T> mean_1 = linalg::mean(select_group(data, groups, group_1));
    Data<T> mean_2 = linalg::mean(select_group(data, groups, group_2));

    T projected_mean_1 = project(mean_1, projector).value();
    T projected_mean_2 = project(mean_2, projector).value();

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

  template std::tuple<int, int> sort_groups_by_threshold(
    Data<double>      data,
    DataColumn<int>   groups,
    int               group_1,
    int               group_2,
    Projector<double> projector,
    Threshold<double> threshold);

  template<typename T, typename R >
  Node<T, R> binary_step(
    Data<T>          data,
    DataColumn<R>    original_groups,
    R                group_1,
    R                group_2,
    PPStrategy<T, R> pp_strategy
    ) {
    std::set<R> unique_groups = { group_1, group_2 };

    auto [projector, projected] = pp_strategy(data, original_groups, unique_groups);

    T threshold = get_threshold(projected, original_groups, group_1, group_2);

    auto [l_group, r_group] = sort_groups_by_threshold(
      data,
      original_groups,
      group_1,
      group_2,
      projector,
      threshold);

    Node<T, R> l_node = { .response = l_group };
    Node<T, R> r_node = { .response = r_group };

    Node<T, R> node = {};
    node.projector = projector;
    node.threshold = threshold;
    node.left = &l_node;
    node.right = &r_node;
    return node;
  }

  template Node<double, int> binary_step(
    Data<double>            data,
    DataColumn<int>         original_groups,
    int                     group_1,
    int                     group_2,
    PPStrategy<double, int> pp_strategy);

  template<typename R >
  std::tuple<R, R> take_two(std::set<R> group_set) {
    return std::make_tuple(*group_set.begin(), *group_set.end());
  }

  template std::tuple<int, int> take_two(std::set<int> group_set);

  template<typename T, typename R >
  Node<T, R> step(
    Data<T>          data,
    DataColumn<R>    original_groups,
    std::set<R>      unique_groups,
    PPStrategy<T, R> pp_strategy) {
    if (unique_groups.size() == 2) {
      auto [group_1, group_2] = take_two(unique_groups);

      return binary_step(
        data,
        original_groups,
        group_1,
        group_2,
        pp_strategy);
    }

    auto [new_groups, new_unique_groups, group_mapping] = as_binary_problem(
      data,
      original_groups,
      unique_groups,
      pp_strategy);

    auto [group_1, group_2] = take_two(new_unique_groups);

    Node<T, R> temp_node = binary_step(
      data,
      new_groups,
      group_1,
      group_2,
      pp_strategy);

    Node<T, R> node = {};
    node.projector = temp_node.projector;
    node.threshold = temp_node.threshold;

    R l_group = temp_node.left->response;
    R r_group = temp_node.right->response;

    *node.left = step(
      select_group(data, new_groups, l_group),
      (DataColumn<R>)select_group((Data<R>)original_groups, new_groups, l_group),
      group_mapping[l_group],
      pp_strategy);

    *node.right = step(
      select_group(data, new_groups, r_group),
      (DataColumn<R>)select_group((Data<R>)original_groups, new_groups, r_group),
      group_mapping[r_group],
      pp_strategy);

    return node;
  };

  template Node<double, int> step(
    Data<double>            data,
    DataColumn<int>         original_groups,
    std::set<int>           unique_groups,
    PPStrategy<double, int> pp_strategy);

  template<typename T, typename R >
  Tree<T, R> train(
    Data<T>          data,
    DataColumn<R>    groups,
    PPStrategy<T, R> pp_strategy) {
    std::set<R> unique_groups = unique(groups);

    Tree<T, R> tree;
    tree.root = step(data, groups, unique_groups, pp_strategy);
    return tree;
  }

  template Tree<double, int> train(
    Data<double>            data,
    DataColumn<int>         groups,
    PPStrategy<double, int> pp_strategy);


  template <typename T, typename R>
  R predict(
    DataColumn<T> data,
    Node<T, R>    node) {
    if (node.left == nullptr && node.right == nullptr) {
      return node.response;
    }

    T projected_data = project((Data<T>)data, node.projector).value();

    if (projected_data < node.threshold) {
      return predict(data, *node.left);
    } else {
      return predict(data, *node.right);
    }
  }

  template int predict(
    DataColumn<double> data,
    Node<double, int>  node);

  template <typename T, typename R>
  R predict(
    DataColumn<T> data,
    Tree<T, R>    tree) {
    return predict(data, tree.root);
  }

  template int predict(
    DataColumn<double> data,
    Tree<double, int>  tree);

  template <typename T, typename R>
  DataColumn<R> predict(
    Data<T>    data,
    Tree<T, R> tree) {
    DataColumn<R> predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((DataColumn<T>)data.row(i), tree);
    }

    return predictions;
  }

  template DataColumn<int> predict(
    Data<double>      data,
    Tree<double, int> tree);
}
