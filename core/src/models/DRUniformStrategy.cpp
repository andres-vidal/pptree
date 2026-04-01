#include "models/DRUniformStrategy.hpp"

#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <numeric>

namespace ppforest2::dr {
  DRUniformStrategy::DRUniformStrategy(int n_vars)
      : n_vars(n_vars) {
    invariant(n_vars > 0, "The number of variables must be greater than 0.");
  }

  void DRUniformStrategy::to_json(nlohmann::json& j) const {
    j = {{"name", "uniform"}, {"n_vars", n_vars}};
  }

  DRResult DRUniformStrategy::select(
      types::FeatureMatrix const& x, stats::GroupPartition const& group_spec, stats::RNG& rng
  ) const {
    invariant(
        n_vars <= x.cols(), "The number of variables must be less than or equal to the number of columns in the data."
    );

    if (n_vars == x.cols()) {
      std::vector<int> all_indices(x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return DRResult(all_indices, x.cols());
    }

    stats::Uniform unif(0, x.cols() - 1);
    std::vector<int> selected_indices = unif.distinct(n_vars, rng);
    return DRResult(selected_indices, x.cols());
  }

  DRStrategy::Ptr uniform(int n_vars) {
    return std::make_shared<DRUniformStrategy>(n_vars);
  }

  DRStrategy::Ptr DRUniformStrategy::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "uniform DR", {"name", "n_vars"});
    return uniform(j.at("n_vars").get<int>());
  }
}
