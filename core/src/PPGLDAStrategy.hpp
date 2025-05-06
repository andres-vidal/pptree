#pragma once

#include "PPStrategy.hpp"

#include <set>
#include <vector>

namespace models::pp::strategy {
  template<typename T, typename G>
  struct PPGLDAStrategy : public PPStrategy<T, G> {
    const float lambda;

    explicit PPGLDAStrategy(const float lambda) : lambda(lambda) {
      if (lambda == 0) {
        LOG_INFO << "Chosen Projection-Pursuit Strategy is LDA" << std::endl;
      } else {
        LOG_INFO << "Chosen Projection-Pursuit Strategy is PDA(lambda = " << lambda << ")" << std::endl;
      }
    }

    PPStrategyPtr<T, G> clone() const override {
      return std::make_unique<PPGLDAStrategy<T, G> >(*this);
    }

    T index(
      const stats::Data<T> &     x,
      const stats::GroupSpec<G>& data_spec,
      const Projector<T>&        projector) const override {
      stats::Data<T> A = projector;

      stats::Data<T> W      = data_spec.wgss(x);
      stats::Data<T> W_diag = W.diagonal().asDiagonal();
      stats::Data<T> W_pda  = W_diag + (1 - lambda) * (W - W_diag);
      stats::Data<T> B      = data_spec.bgss(x);
      stats::Data<T> WpB    = W_pda + B;

      T denominator = (A.transpose() * WpB * A).determinant();

      if (fabs(denominator) < 1e-15) {
        return 0;
      }

      T numerator = (A.transpose() * W_pda * A).determinant();

      return 1 - numerator / denominator;
    }

    Projector<T> optimize(const stats::Data<T> &x, const stats::GroupSpec<G>& data_spec) const override {
      LOG_INFO << "Calculating PDA optimum projector for " << data_spec.groups.size() << " groups: " << data_spec.groups << std::endl;
      LOG_INFO << "Dataset size: " << data_spec.rows(x) << " observations of " << x.cols() << " variables:" << std::endl;
      LOG_INFO << std::endl << (stats::Data<T>)data_spec.data(x) << std::endl;

      stats::Data<T> B = data_spec.bgss(x);
      stats::Data<T> W = data_spec.wgss(x);

      LOG_INFO << "B:" << std::endl << B << std::endl;
      LOG_INFO << "W:" << std::endl << W << std::endl;

      stats::Data<T> W_pda = (1 - lambda) * W;
      W_pda.diagonal() = W.diagonal();

      stats::Data<T> WpB = W_pda + B;

      LOG_INFO << "W_pda:" << std::endl << W_pda << std::endl;
      LOG_INFO << "W_pda + B:" << std::endl << WpB << std::endl;

      stats::Data<T> WpBInvB = WpB.fullPivLu().solve(B);

      LOG_INFO << "(W_pda + B)^-1 * B:" << std::endl << WpBInvB << std::endl;

      Eigen::EigenSolver<math::DMatrix<T> > eigen_solver(WpBInvB);
      math::DVector<T> eigen_val = eigen_solver.eigenvalues().real();
      math::DMatrix<T> eigen_vec = eigen_solver.eigenvectors().real();

      LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
      LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

      std::vector<Eigen::Index> indices(eigen_val.size());
      std::iota(indices.begin(), indices.end(), 0);

      std::sort(indices.begin(), indices.end(), [&eigen_val, &eigen_vec](Eigen::Index idx1, Eigen::Index idx2) {
          float value1_mod = fabs(eigen_val(idx1));
          float value2_mod = fabs(eigen_val(idx2));

          if (math::is_approx(value1_mod, value2_mod, 0.001)) {
            stats::DataColumn<T> vector1 = eigen_vec.col(idx1);
            stats::DataColumn<T> vector2 = eigen_vec.col(idx2);

            for (int i = 0; i < vector1.size(); ++i) {
              if (!math::is_module_approx(vector1[i], vector2[i]) ) {
                return vector1[i] < vector2[i];
              }
            }
          }

          return value1_mod < value2_mod;
        });

      math::DVector<T> max_eigen_vec = eigen_vec.col(indices.back());

      LOG_INFO << "Maximal eigenvector:" << std::endl << max_eigen_vec << std::endl;

      Projector<T> projector = pp::normalize(max_eigen_vec);

      LOG_INFO << "Projector:" << std::endl << projector << std::endl;
      return projector;
    }

    static PPStrategyPtr<T, G> make(const float lambda) {
      return std::make_unique<PPGLDAStrategy<T, G> >(lambda);
    }
  };

  template<typename T, typename G>
  PPStrategyPtr<T, G> glda(const float lambda) {
    return PPGLDAStrategy<T, G>::make(lambda);
  }
}
