#include "models/SRMeanOfMeansStrategy.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::sr {
  void SRMeanOfMeansStrategy::to_json(nlohmann::json& j) const {
    j = { { "name", "mean_of_means" } };
  }

  types::Feature SRMeanOfMeansStrategy::threshold(
    const types::FeatureMatrix& group_1,
    const types::FeatureMatrix& group_2,
    const pp::Projector&        projector) const {
    return ((group_1 * projector).mean() + (group_2 * projector).mean()) / 2;
  }

  SRStrategy::Ptr mean_of_means() {
    return std::make_shared<SRMeanOfMeansStrategy>();
  }

  SRStrategy::Ptr SRMeanOfMeansStrategy::from_json(const nlohmann::json& j) {
    validate_json_keys(j, "mean_of_means SR", { "name" });
    return mean_of_means();
  }
}
