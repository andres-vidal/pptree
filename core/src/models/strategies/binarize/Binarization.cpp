#include "models/strategies/binarize/Binarization.hpp"

#include "models/strategies/NodeContext.hpp"

namespace ppforest2::binarize {
  void Binarization::operator()(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }
    regroup(ctx, rng);
  }
}
