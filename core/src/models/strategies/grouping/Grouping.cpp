#include "models/strategies/grouping/Grouping.hpp"

#include "models/strategies/NodeContext.hpp"

namespace ppforest2::grouping {
  void Grouping::operator()(NodeContext& ctx, types::GroupId lower, types::GroupId upper, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }
    split(ctx, lower, upper, rng);
  }
}
