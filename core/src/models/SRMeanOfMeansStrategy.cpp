#include "models/SRMeanOfMeansStrategy.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::sr {
  void SRMeanOfMeansStrategy::to_json(nlohmann::json& j) const {
    j = {{"name", "mean_of_means"}};
  }

  types::Feature SRMeanOfMeansStrategy::threshold(
      types::FeatureMatrix const& group_1, types::FeatureMatrix const& group_2, pp::Projector const& projector
  ) const {
    return ((group_1 * projector).mean() + (group_2 * projector).mean()) / 2;
  }

  SRStrategy::Ptr mean_of_means() {
    return std::make_shared<SRMeanOfMeansStrategy>();
  }

  SRStrategy::Ptr SRMeanOfMeansStrategy::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "mean_of_means SR", {"name"});
    return mean_of_means();
  }
}
