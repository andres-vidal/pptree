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

  void ByLabel::split(NodeContext& ctx, GroupId lower, GroupId upper, RNG& /*rng*/) const {
    auto [lower_y_part, upper_y_part] = compute(ctx.active_partition(), lower, upper);
    ctx.lower_y_part.emplace(std::move(lower_y_part));
    ctx.upper_y_part.emplace(std::move(upper_y_part));
  }

  std::pair<GroupPartition, GroupPartition>
  ByLabel::compute(GroupPartition const& y_part, GroupId lower, GroupId upper) const {
    auto lower_y_part = y_part.subset(y_part.subgroups.at(lower));
    auto upper_y_part = y_part.subset(y_part.subgroups.at(upper));

    return {std::move(lower_y_part), std::move(upper_y_part)};
  }

  Grouping::Ptr by_label() {
    return std::make_shared<ByLabel>();
  }

  Grouping::Ptr ByLabel::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_label"}.only_keys({"name"});
    return by_label();
  }
}
