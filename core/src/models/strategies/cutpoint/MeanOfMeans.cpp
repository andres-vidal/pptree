#include "models/strategies/cutpoint/MeanOfMeans.hpp"

#include "models/strategies/NodeContext.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::cutpoint {
  nlohmann::json MeanOfMeans::to_json() const {
    return {{"name", "mean_of_means"}};
  }

  void MeanOfMeans::cutpoint(NodeContext& ctx, RNG& /*rng*/) const {
    auto const& partition = ctx.active_partition();
    auto g1               = *partition.groups.begin();
    auto g2               = *std::next(partition.groups.begin());
    auto data_1           = partition.group(ctx.x, g1);
    auto data_2           = partition.group(ctx.x, g2);
    ctx.cutpoint          = compute(data_1, data_2, ctx.projector);
  }

  Feature MeanOfMeans::compute(
      FeatureMatrix const& group_1, FeatureMatrix const& group_2, ppforest2::pp::Projector const& projector
  ) const {
    return ((group_1 * projector).mean() + (group_2 * projector).mean()) / 2;
  }

  SplitCutpoint::Ptr mean_of_means() {
    return std::make_shared<MeanOfMeans>();
  }

  SplitCutpoint::Ptr MeanOfMeans::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "mean_of_means cutpoint", {"name"});
    return mean_of_means();
  }
}
