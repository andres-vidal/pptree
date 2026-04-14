#include "models/strategies/vars/Uniform.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <numeric>

namespace ppforest2::vars {
  Uniform::Uniform(int n_vars)
      : n_vars(n_vars) {
    invariant(n_vars > 0, "The number of variables must be greater than 0.");
  }

  nlohmann::json Uniform::to_json() const {
    return {{"name", "uniform"}, {"count", n_vars}};
  }

  void Uniform::select(NodeContext& ctx, stats::RNG& rng) const {
    ctx.var_selection = compute(ctx.x, rng);
  }

  VariableSelection::Result Uniform::compute(types::FeatureMatrix const& x, stats::RNG& rng) const {
    invariant(
        n_vars <= x.cols(), "The number of variables must be less than or equal to the number of columns in the data."
    );

    if (n_vars == x.cols()) {
      std::vector<int> all_indices(x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return VariableSelection::Result(all_indices, x.cols());
    }

    stats::Uniform unif(0, x.cols() - 1);

    return VariableSelection::Result(unif.distinct(n_vars, rng), x.cols());
  }

  VariableSelection::Ptr uniform(int n_vars) {
    return std::make_shared<Uniform>(n_vars);
  }

  VariableSelection::Ptr Uniform::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "uniform vars", {"name", "count", "proportion"});

    if (j.contains("proportion")) {
      float p = j.at("proportion").get<float>();
      invariant(p > 0 && p <= 1, "proportion must be in (0, 1]");
      // Proportion is resolved to count later when total_vars is known.
      // Return a placeholder — caller must resolve before use.
      return uniform(1);
    }

    return uniform(j.at("count").get<int>());
  }
}
