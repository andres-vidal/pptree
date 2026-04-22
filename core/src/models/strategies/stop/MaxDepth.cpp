#include "models/strategies/stop/MaxDepth.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::stop {
  MaxDepth::MaxDepth(int max_depth)
      : max_depth(max_depth) {
    // `max_depth < 0` produces a rule that fires at the root (depth 0 >=
    // max_depth for any negative max), i.e. the tree never splits — a
    // degenerate configuration that's almost certainly a user mistake
    // rather than an intentional "stump-only" request. Rejected here so
    // the error surfaces at construction rather than as a mysterious
    // all-leaves model. `max_depth == 0` is accepted and explicitly
    // produces a stump (root is a leaf); unusual but occasionally useful
    // for debugging or minimum-complexity baselines.
    user_error(
        max_depth >= 0,
        "stop_max_depth: max_depth must be >= 0 (got " + std::to_string(max_depth) +
            "). Use 0 for a root-only stump, or a positive integer for a depth-bounded tree."
    );
  }

  nlohmann::json MaxDepth::to_json() const {
    return {{"name", "max_depth"}, {"max_depth", max_depth}};
  }

  std::string MaxDepth::display_name() const {
    return "Max depth (" + std::to_string(max_depth) + ")";
  }

  bool MaxDepth::should_stop(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    return ctx.depth >= max_depth;
  }

  StopRule::Ptr max_depth(int n) {
    return std::make_shared<MaxDepth>(n);
  }

  StopRule::Ptr MaxDepth::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "max_depth"};
    r.only_keys({"name", "max_depth"});
    return stop::max_depth(static_cast<int>(r.require_int("max_depth", 0)));
  }
}
