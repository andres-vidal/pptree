#include "models/PPPDAStrategy.hpp"

#include "utils/Math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;


namespace ppforest2::pp {
  namespace {
    PPResult nan_result(FeatureMatrix const& x) {
      return PPResult{
          FeatureVector::Constant(x.cols(), std::numeric_limits<Feature>::quiet_NaN()),
          std::numeric_limits<Feature>::quiet_NaN()
      };
    }
  }

  PPPDAStrategy::PPPDAStrategy(float lambda)
      : lambda(lambda) {}

  void PPPDAStrategy::to_json(nlohmann::json& j) const {
    j = {{"name", "pda"}, {"lambda", lambda}};
  }

  Feature
  PPPDAStrategy::index(FeatureMatrix const& x, GroupPartition const& group_spec, Projector const& projector) const {
    FeatureMatrix A = projector;

    FeatureMatrix W      = group_spec.wgss(x);
    FeatureMatrix W_diag = W.diagonal().asDiagonal();
    FeatureMatrix W_pda  = W_diag + (1 - lambda) * (W - W_diag);
    FeatureMatrix B      = group_spec.bgss(x);
    FeatureMatrix WpB    = W_pda + B;

    Feature denominator = (A.transpose() * WpB * A).determinant();

    if (std::fabs(denominator) < 1e-15) {
      return Feature(0);
    }

    Feature numerator = (A.transpose() * W_pda * A).determinant();

    return Feature(1) - numerator / denominator;
  }

  PPResult PPPDAStrategy::optimize(FeatureMatrix const& x, GroupPartition const& group_spec) const {
    FeatureMatrix const B = group_spec.bgss(x);
    FeatureMatrix const W = group_spec.wgss(x);

    FeatureMatrix W_pda = (Feature(1) - Feature(lambda)) * W;
    W_pda.diagonal()    = W.diagonal();

    FeatureMatrix const WpB = W_pda + B; // symmetric

    Eigen::GeneralizedSelfAdjointEigenSolver<FeatureMatrix> ges;
    ges.compute(B, WpB);

    if (ges.info() != Eigen::Success)
      return nan_result(x);

    // largest eigenvalue => best 1D projection
    FeatureVector max_eigen_vec = ges.eigenvectors().col(ges.eigenvalues().size() - 1);
    Feature max_eigen_val       = ges.eigenvalues().real().maxCoeff();

    // Solver may report Success but produce NaN eigenvectors when B is
    // positive-semidefinite (singular covariance from small bootstrap samples).
    if (max_eigen_vec.hasNaN() || std::isnan(max_eigen_val))
      return nan_result(x);

    return PPResult{pp::normalize(max_eigen_vec), max_eigen_val};
  }

  PPStrategy::Ptr pda(float lambda) {
    return std::make_shared<PPPDAStrategy>(lambda);
  }

  PPStrategy::Ptr PPPDAStrategy::from_json(nlohmann::json const& j) {
    validate_json_keys(j, "PDA", {"name", "lambda"});
    return pda(j.at("lambda").get<float>());
  }
}
