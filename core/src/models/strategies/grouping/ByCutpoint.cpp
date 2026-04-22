#include "models/strategies/grouping/ByCutpoint.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::grouping {
  nlohmann::json ByCutpoint::to_json() const {
    return {{"name", "by_cutpoint"}};
  }

  GroupPartition ByCutpoint::init(OutcomeVector const& y) const {
    int const n = static_cast<int>(y.size());
    invariant(n > 0, "ByCutpoint::init requires a non-empty response vector");

    if (n < 2) {
      return GroupPartition::single_group(0, n - 1);
    }

    int const mid = n / 2;
    return GroupPartition::two_groups(0, mid - 1, mid, n - 1);
  }

  /**
   * @brief Partition rows in [start, end] by projected value vs cutpoint.
   *
   * Reorders x and continuous_y in place so that rows with projected
   * value < cutpoint come first. Returns the index of the first right-child
   * row (i.e., left child owns [start, mid-1], right owns [mid, end]).
   */
  static int partition_by_cutpoint(
      FeatureMatrix& x,
      OutcomeVector& continuous_y,
      pp::Projector const& projector,
      Feature cutpoint,
      int start,
      int end
  ) {
    int const n = end - start + 1;
    FeatureVector projected(n);

    for (int i = 0; i < n; ++i) {
      projected(i) = x.row(start + i).dot(projector);
    }

    // Build index array and partition: left (projected < cutpoint) first.
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    auto pivot = std::stable_partition(order.begin(), order.end(), [&](int i) { return projected(i) < cutpoint; });

    int const left_count = static_cast<int>(std::distance(order.begin(), pivot));

    // Apply permutation to x and continuous_y.
    FeatureMatrix x_tmp = x.middleRows(start, n);
    OutcomeVector y_tmp = continuous_y.segment(start, n);

    for (int i = 0; i < n; ++i) {
      x.row(start + i)        = x_tmp.row(order[static_cast<std::size_t>(i)]);
      continuous_y(start + i) = y_tmp(order[static_cast<std::size_t>(i)]);
    }

    return start + left_count;
  }

  /**
   * @brief Sort rows in [start, end] by continuous_y value (ascending).
   *
   * Reorders x and continuous_y in place.
   */
  static void sort_by_continuous_y(FeatureMatrix& x, OutcomeVector& continuous_y, int start, int end) {
    int const n = end - start + 1;

    if (n <= 1) {
      return;
    }

    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return continuous_y(start + a) < continuous_y(start + b);
    });

    FeatureMatrix x_tmp = x.middleRows(start, n);
    OutcomeVector y_tmp = continuous_y.segment(start, n);

    for (int i = 0; i < n; ++i) {
      x.row(start + i)        = x_tmp.row(order[static_cast<std::size_t>(i)]);
      continuous_y(start + i) = y_tmp(order[static_cast<std::size_t>(i)]);
    }
  }

  /**
   * @brief Create a 2-group partition by median-splitting rows [start, end].
   *
   * Assumes rows are already sorted by continuous_y within this range.
   * Group 0 gets the first half, group 1 gets the second half.
   *
   * Requires n >= 2 (caller must check).
   */
  static GroupPartition median_split(int start, int end) {
    int const n   = end - start + 1;
    int const mid = n / 2;

    invariant(n >= 2, "median_split requires at least 2 observations");

    return GroupPartition::two_groups(start, start + mid - 1, start + mid, end);
  }

  void ByCutpoint::split(NodeContext& ctx, GroupId lower, GroupId upper, stats::RNG& /*rng*/) const {
    invariant(ctx.y_vec != nullptr, "ByCutpoint requires y_vec on NodeContext");
    invariant(ctx.projector.has_value(), "ByCutpoint requires projector on NodeContext");
    invariant(ctx.cutpoint.has_value(), "ByCutpoint requires cutpoint on NodeContext");

    auto const& y_part = ctx.active_partition();

    // Get the contiguous range this node owns.
    int const node_start = std::min(y_part.group_start(lower), y_part.group_start(upper));
    int const node_end   = std::max(y_part.group_end(lower), y_part.group_end(upper));

    // 1. Partition by cutpoint: left-bound rows first.
    int const mid = partition_by_cutpoint(ctx.x, *ctx.y_vec, *ctx.projector, *ctx.cutpoint, node_start, node_end);

    int const left_n  = mid - node_start;
    int const right_n = node_end - mid + 1;

    // Edge case: all observations go to one side of the cutpoint. Write
    // identical partitions for both children; the tree builder's no-progress
    // guard (Tree::build_root) detects this and converts the node to a leaf.
    if (left_n == 0) {
      sort_by_continuous_y(ctx.x, *ctx.y_vec, mid, node_end);
      auto edge_y_part = right_n >= 2 ? median_split(mid, node_end) : GroupPartition::single_group(mid, node_end);
      ctx.lower_y_part.emplace(edge_y_part);
      ctx.upper_y_part.emplace(edge_y_part);
      return;
    }

    if (right_n == 0) {
      sort_by_continuous_y(ctx.x, *ctx.y_vec, node_start, node_end);
      auto edge_y_part =
          left_n >= 2 ? median_split(node_start, node_end) : GroupPartition::single_group(node_start, node_end);
      ctx.lower_y_part.emplace(edge_y_part);
      ctx.upper_y_part.emplace(edge_y_part);
      return;
    }

    // 2. Sort each child's range by continuous_y.
    sort_by_continuous_y(ctx.x, *ctx.y_vec, node_start, mid - 1);
    sort_by_continuous_y(ctx.x, *ctx.y_vec, mid, node_end);

    // 3. Median-split each child into 2 groups (if enough observations).
    // Children with < 2 observations get a single-group partition;
    // the stop rule will turn them into leaves.
    auto make_child_partition = [](int start, int end) -> GroupPartition {
      int const n = end - start + 1;

      if (n < 2) {
        return GroupPartition::single_group(start, end);
      }

      return median_split(start, end);
    };

    ctx.lower_y_part.emplace(make_child_partition(node_start, mid - 1));
    ctx.upper_y_part.emplace(make_child_partition(mid, node_end));
  }

  Grouping::Ptr by_cutpoint() {
    return std::make_shared<ByCutpoint>();
  }

  Grouping::Ptr ByCutpoint::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_cutpoint"}.only_keys({"name"});
    return by_cutpoint();
  }
}
