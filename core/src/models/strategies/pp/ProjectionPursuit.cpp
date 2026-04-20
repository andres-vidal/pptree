#include "models/strategies/pp/ProjectionPursuit.hpp"

#include "models/strategies/NodeContext.hpp"

namespace ppforest2::pp {
  void ProjectionPursuit::operator()(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }
    optimize(ctx, rng);
  }
}
