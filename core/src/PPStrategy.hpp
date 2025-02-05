#pragma once

#include "Projector.hpp"
#include "SortedDataSpec.hpp"

#include <set>
#include <vector>

namespace models::pp::strategy {
  template<typename T, typename G>
  struct PPStrategy {
    virtual ~PPStrategy() = default;
    virtual std::unique_ptr<PPStrategy<T, G> > clone() const = 0;

    virtual T index(
      const stats::SortedDataSpec<T, G>& data,
      const Projector<T>&                projector) const = 0;

    virtual Projector<T> optimize(const stats::SortedDataSpec<T, G>& data) const = 0;

    Projector<T> operator()(const stats::SortedDataSpec<T, G>& data) const {
      return optimize(data);
    }
  };

  template<typename T, typename G>
  struct GLDAStrategy : public PPStrategy<T, G> {
    const float lambda;

    explicit GLDAStrategy(const float lambda) : lambda(lambda) {
      if (lambda == 0) {
        LOG_INFO << "Chosen Projection-Pursuit Strategy is LDA" << std::endl;
      } else {
        LOG_INFO << "Chosen Projection-Pursuit Strategy is PDA(lambda = " << lambda << ")" << std::endl;
      }
    }

    std::unique_ptr<PPStrategy<T, G> > clone() const override {
      return std::make_unique<GLDAStrategy<T, G> >(*this);
    }

    T index(
      const stats::SortedDataSpec<T, G>& data,
      const Projector<T>&                projector) const override {
      stats::Data<T> A = projector;

      stats::Data<T> W = data.wgss();
      stats::Data<T> W_diag = W.diagonal().asDiagonal();
      stats::Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
      stats::Data<T> B = data.bgss();
      stats::Data<T> WpB = W_pda + B;

      T denominator = math::determinant(math::inner_square(A, WpB));

      if (denominator == 0) {
        return 0;
      }

      return 1 - math::determinant(math::inner_square(A, W_pda)) / denominator;
    }

    Projector<T> optimize(const stats::SortedDataSpec<T, G>& data) const override {
      LOG_INFO << "Calculating PDA optimum projector for " << data.classes.size() << " groups: " << data.classes << std::endl;
      LOG_INFO << "Dataset size: " << data.x.rows() << " observations of " << data.x.cols() << " variables:" << std::endl;
      LOG_INFO << std::endl << data.x << std::endl;
      LOG_INFO << "Groups:" << std::endl;
      LOG_INFO << std::endl << data.y << std::endl;

      auto B = data.bgss();
      auto W = data.wgss();

      LOG_INFO << "B:" << std::endl << B << std::endl;
      LOG_INFO << "W:" << std::endl << W << std::endl;

      auto W_diag = W.diagonal().asDiagonal().toDenseMatrix();
      auto W_pda = W_diag + (1 - lambda) * (W - W_diag);
      auto WpB = W_pda + B;

      LOG_INFO << "W_pda:" << std::endl << W_pda << std::endl;
      LOG_INFO << "W_pda + B:" << std::endl << WpB << std::endl;

      stats::Data<T> WpBInvB = WpB.fullPivLu().solve(B);
      stats::Data<T> truncatedWpBInvB = math::truncate(WpBInvB);

      LOG_INFO << "(W_pda + B)^-1 * B:" << std::endl << WpBInvB << std::endl;
      LOG_INFO << "(W_pda + B)^-1 * B (truncated):" << std::endl << truncatedWpBInvB << std::endl;

      Eigen::EigenSolver<math::DMatrix<T> > eigen_solver(truncatedWpBInvB);
      math::DVector<T> eigen_val = eigen_solver.eigenvalues().real();
      math::DMatrix<T> eigen_vec = eigen_solver.eigenvectors().real();

      LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
      LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

      std::vector<Eigen::Index> indices(eigen_val.size());
      std::iota(indices.begin(), indices.end(), 0);

      std::sort(indices.begin(), indices.end(), [&eigen_val, &eigen_vec](Eigen::Index idx1, Eigen::Index idx2) {
         float value1_mod = fabs(eigen_val.row(idx1).value());
         float value2_mod = fabs(eigen_val.row(idx2).value());

         if (math::is_approx(value1_mod, value2_mod, 0.001)) {
           auto vector1 = eigen_vec.col(idx1);
           auto vector2 = eigen_vec.col(idx2);

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
  };

  template<typename T, typename G>
  std::unique_ptr<PPStrategy<T, G> > glda(const float lambda) {
    return std::make_unique<GLDAStrategy<T, G> >(lambda);
  }
}
