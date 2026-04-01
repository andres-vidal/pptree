#include "models/DRNoopStrategy.hpp"

#include <numeric>

#include <nlohmann/json.hpp>

namespace ppforest2::dr {
  void DRNoopStrategy::to_json(nlohmann::json& j) const {
    j = {{"name", "noop"}};
  }

  DRResult DRNoopStrategy::select(types::FeatureMatrix const& x,
                                  stats::GroupPartition const& group_spec,
                                  stats::RNG& rng) const {
    std::vector<int> all_indices(x.cols());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    return DRResult(all_indices, x.cols());
  }

  DRStrategy::Ptr noop() {
    return std::make_shared<DRNoopStrategy>();
  }

  DRStrategy::Ptr DRNoopStrategy::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "noop DR", {"name"});
    return noop();
  }
}
