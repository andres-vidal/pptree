#include "models/strategies/binarize/LargestGap.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::binarize {
  nlohmann::json LargestGap::to_json() const {
    return {{"name", "largest_gap"}};
  }

  void LargestGap::regroup(NodeContext& ctx, RNG& /*rng*/) const {
    FeatureMatrix const projected_x = ctx.x * ctx.projector;
    auto result                     = compute(projected_x, ctx.y);
    ctx.binary_y.emplace(std::move(result.binary_y));
    ctx.binary_0 = result.group_0;
    ctx.binary_1 = result.group_1;
  }

  Binarization::Result LargestGap::compute(FeatureMatrix const& projected_x, GroupPartition const& y) const {
    std::vector<std::tuple<Outcome, Feature>> means;

    invariant(projected_x.cols() == 1, "Binary regrouping requires unidimensional data");

    for (Outcome const group : y.groups) {
      Feature const group_mean = y.group(projected_x, group).mean();
      means.emplace_back(group, group_mean);
    }

    std::stable_sort(means.begin(), means.end(), [](auto const& a, auto const& b) {
      return std::isless(std::get<1>(a), std::get<1>(b));
    });

    Feature edge_gap   = -1;
    Outcome edge_group = -1;

    for (size_t i = 0; i + 1 < means.size(); i++) {
      Feature const gap = std::get<1>(means[i + 1]) - std::get<1>(means[i]);

      if (gap > edge_gap) {
        edge_gap   = gap;
        edge_group = std::get<0>(means[i + 1]);
      }
    }

    if (edge_group == -1) {
      edge_group = std::get<0>(means.front());
    }

    std::map<Outcome, int> binary_mapping;

    bool edge_found = false;
    for (auto const& [group, mean] : means) {
      edge_found            = edge_found || group == edge_group;
      binary_mapping[group] = edge_found ? 1 : 0;
    }

    auto binary_y = y.remap(binary_mapping);

    Outcome const group_0 = *binary_y.groups.begin();
    Outcome const group_1 = *std::next(binary_y.groups.begin());

    return Binarization::Result{std::move(binary_y), group_0, group_1};
  }

  Binarization::Ptr largest_gap() {
    return std::make_shared<LargestGap>();
  }

  Binarization::Ptr LargestGap::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "largest_gap binarize", {"name"});
    return largest_gap();
  }
}
