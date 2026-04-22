#include "models/strategies/vars/VariableSelection.hpp"

#include "models/strategies/NodeContext.hpp"

namespace ppforest2::vars {
  void VariableSelection::operator()(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }
    select(ctx, rng);
  }
}
