#include "models/strategies/vars/All.hpp"

#include "models/strategies/NodeContext.hpp"

#include <numeric>

#include <nlohmann/json.hpp>

namespace ppforest2::vars {
  nlohmann::json All::to_json() const {
    return {{"name", "all"}};
  }

  void All::select(NodeContext& ctx, stats::RNG& /*rng*/) const {
    ctx.var_selection = compute(ctx.x);
  }

  VariableSelection::Result All::compute(types::FeatureMatrix const& x) const {
    std::vector<int> all_indices(x.cols());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    return VariableSelection::Result(all_indices, x.cols());
  }

  VariableSelection::Ptr all() {
    return std::make_shared<All>();
  }

  VariableSelection::Ptr All::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "all vars", {"name"});
    return all();
  }
}
