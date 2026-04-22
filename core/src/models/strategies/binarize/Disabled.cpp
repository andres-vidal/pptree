#include "models/strategies/binarize/Disabled.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::binarize {
  nlohmann::json Disabled::to_json() const {
    return {{"name", "disabled"}};
  }

  void Disabled::regroup(NodeContext& /*ctx*/, stats::RNG& /*rng*/) const {
    // Unconditional invariant: Disabled is only ever correct in specs
    // whose grouping guarantees a ≤2-group partition at every node,
    // so the binary path in tree training doesn't invoke binarize. If
    // we reach here the spec is inconsistent (grouping produced >2
    // groups with Disabled configured). Fire an invariant with a
    // clear message rather than silently passing through or producing
    // a confusing downstream assertion.
    invariant(
        false,
        "binarize::Disabled was invoked. This placeholder binarizer "
        "should never fire: either the grouping strategy produced a "
        ">2-group partition (use binarize::largest_gap or another real "
        "binarizer instead), or the spec was assembled incorrectly."
    );
  }

  Binarization::Ptr disabled() {
    return std::make_shared<Disabled>();
  }

  Binarization::Ptr Disabled::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "disabled"};
    r.only_keys({"name"});
    return disabled();
  }
}
