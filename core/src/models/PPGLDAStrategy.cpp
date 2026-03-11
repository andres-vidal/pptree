#include "models/PPGLDAStrategy.hpp"

#include "utils/Math.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

using namespace pptree::types;
using namespace pptree::stats;


namespace pptree::pp {
  PPGLDAStrategy::PPGLDAStrategy(float lambda) :
    lambda(lambda) {
  }

  PPStrategy::Ptr PPGLDAStrategy::clone() const {
    return std::make_unique<PPGLDAStrategy>(*this);
  }

  Feature PPGLDAStrategy::index(
    FeatureMatrix const&  x,
    GroupPartition const& group_spec,
    Projector const&      projector) const {
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

  PPResult PPGLDAStrategy::optimize(
    const FeatureMatrix&  x,
    const GroupPartition& group_spec) const {
    FeatureMatrix B = group_spec.bgss(x);
    FeatureMatrix W = group_spec.wgss(x);

    FeatureMatrix W_pda = (1 - lambda) * W;
    W_pda.diagonal() = W.diagonal();

    FeatureMatrix WpB = W_pda + B;

    FeatureMatrix WpBInvB = WpB.fullPivLu().solve(B);

    Eigen::EigenSolver<FeatureMatrix> eigen_solver(WpBInvB);
    FeatureVector eigen_val = eigen_solver.eigenvalues().real();
    FeatureMatrix eigen_vec = eigen_solver.eigenvectors().real();

    std::vector<Eigen::Index> indices(static_cast<std::size_t>(eigen_val.size()));
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
      [&eigen_val, &eigen_vec](Eigen::Index idx1, Eigen::Index idx2) {
        float value1_mod = std::fabs(eigen_val(idx1));
        float value2_mod = std::fabs(eigen_val(idx2));

        if (math::is_approx(value1_mod, value2_mod, 0.001f)) {
          FeatureVector vector1 = eigen_vec.col(idx1);
          FeatureVector vector2 = eigen_vec.col(idx2);

          for (Eigen::Index i = 0; i < vector1.size(); ++i) {
            if (!math::is_module_approx(vector1[i], vector2[i])) {
              return vector1[i] < vector2[i];
            }
          }
        }

        return value1_mod < value2_mod;
      });

    FeatureVector max_eigen_vec = eigen_vec.col(indices.back());
    Feature max_eigen_val       = eigen_val(indices.back());

    return PPResult{ pp::normalize(max_eigen_vec), max_eigen_val };
  }

  PPStrategy::Ptr PPGLDAStrategy::make(float lambda) {
    return std::make_unique<PPGLDAStrategy>(lambda);
  }

  PPStrategy::Ptr glda(float lambda) {
    return PPGLDAStrategy::make(lambda);
  }
}
