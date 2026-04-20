#include "models/strategies/cutpoint/Cutpoint.hpp"

#include "models/strategies/NodeContext.hpp"

namespace ppforest2::cutpoint {
  void Cutpoint::operator()(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }
    cutpoint(ctx, rng);
  }
}
