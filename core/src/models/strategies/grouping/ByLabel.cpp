#include "models/strategies/grouping/ByLabel.hpp"

#include "models/strategies/NodeContext.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::grouping {
  nlohmann::json ByLabel::to_json() const {
    return {{"name", "by_label"}};
  }

  GroupPartition ByLabel::init(OutcomeVector const& y) const {
    return GroupPartition(y);
  }

  Grouping::Result ByLabel::split(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.binary_y.has_value(), "Binary partition not available");
    return compute(ctx.binary_y.value(), ctx.binary_0, ctx.binary_1);
  }

  Grouping::Result ByLabel::compute(GroupPartition const& binary_y, GroupId group_0, GroupId group_1) const {
    auto lower_y = binary_y.subset(binary_y.subgroups.at(group_0));
    auto upper_y = binary_y.subset(binary_y.subgroups.at(group_1));

    return Grouping::Result{std::move(lower_y), std::move(upper_y)};
  }

  Grouping::Ptr by_label() {
    return std::make_shared<ByLabel>();
  }

  Grouping::Ptr ByLabel::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_label"}.only_keys({"name"});
    return by_label();
  }
}
