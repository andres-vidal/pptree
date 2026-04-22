#include "models/strategies/leaf/MeanResponse.hpp"

#include "models/TreeLeaf.hpp"
#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::leaf {
  nlohmann::json MeanResponse::to_json() const {
    return {{"name", "mean_response"}};
  }

  TreeNode::Ptr MeanResponse::create_leaf(NodeContext const& ctx, RNG& /*rng*/) const {
    invariant(ctx.y_vec != nullptr, "MeanResponse requires y_vec in NodeContext");

    // Subset of y for observations in this node, evaluated in double precision
    // to avoid float summation reorderings under -O2. Eigen's reduction order
    // is determined by the build-time SIMD baseline (the project builds
    // without -march=native, so SIMD width is consistent across platforms),
    // keeping the mean cross-platform reproducible.
    Eigen::VectorXd const y = ctx.y.data(*ctx.y_vec).template cast<double>();

    invariant(y.size() > 0, "MeanResponse: empty node (no observations in any group)");

    return TreeLeaf::make(static_cast<Feature>(y.mean()));
  }

  LeafStrategy::Ptr mean_response() {
    return std::make_shared<MeanResponse>();
  }

  LeafStrategy::Ptr MeanResponse::from_json(nlohmann::json const& j) {
    JsonReader{j, "mean_response"}.only_keys({"name"});
    return mean_response();
  }
}
