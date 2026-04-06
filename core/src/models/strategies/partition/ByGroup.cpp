#include "models/strategies/partition/ByGroup.hpp"

#include "models/strategies/NodeContext.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::partition {
  nlohmann::json ByGroup::to_json() const {
    return {{"name", "by_group"}};
  }

  StepPartition::Result ByGroup::split(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.binary_y.has_value(), "Binary partition not available");
    return compute(ctx.binary_y.value(), ctx.binary_0, ctx.binary_1);
  }

  StepPartition::Result ByGroup::compute(GroupPartition const& binary_y, Outcome group_0, Outcome group_1) const {
    auto lower_y = binary_y.subset(binary_y.subgroups.at(group_0));
    auto upper_y = binary_y.subset(binary_y.subgroups.at(group_1));

    return StepPartition::Result{std::move(lower_y), std::move(upper_y)};
  }

  StepPartition::Ptr by_group() {
    return std::make_shared<ByGroup>();
  }

  StepPartition::Ptr ByGroup::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "by_group partition", {"name"});
    return by_group();
  }
}
