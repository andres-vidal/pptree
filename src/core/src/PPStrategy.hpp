#pragma once

#include "Projector.hpp"

#include <set>

namespace models::pp::strategy {
  template<typename T, typename G>
  struct PPStrategy {
    virtual ~PPStrategy() = default;
    virtual std::unique_ptr<PPStrategy<T, G> > clone() const = 0;

    virtual T index(
      const stats::Data<T>&       data,
      const Projector<T>&         projector,
      const stats::DataColumn<G>& groups,
      const std::set<G>&          unique_groups) const = 0;

    virtual Projector<T> optimize(
      const stats::Data<T>&       data,
      const stats::DataColumn<G>& groups,
      const std::set<G>&          unique_groups) const = 0;

    std::tuple<Projector<T>, Projection<T> > operator()(
      const stats::Data<T>&       data,
      const stats::DataColumn<G>& groups,
      const std::set<G>&          unique_groups
      ) const {
      Projector<T> projector = optimize(data, groups, unique_groups);
      Projection<T> projection = project(data, projector);
      return { projector, projection };
    }
  };

  template<typename T, typename G>
  struct GLDAStrategy : public PPStrategy<T, G> {
    const double lambda;

    explicit GLDAStrategy(const double lambda) : lambda(lambda) {
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
      const stats::Data<T>&       data,
      const Projector<T>&         projector,
      const stats::DataColumn<G>& groups,
      const std::set<G>&          unique_groups) const override {
      stats::Data<T> A = projector;

      stats::Data<T> W = stats::within_groups_sum_of_squares(data, groups, unique_groups);
      stats::Data<T> W_diag = W.diagonal().asDiagonal();
      stats::Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
      stats::Data<T> B = stats::between_groups_sum_of_squares(data, groups, unique_groups);
      stats::Data<T> WpB = W_pda + B;

      T denominator = math::determinant(math::inner_square(A, WpB));

      if (denominator == 0) {
        return 0;
      }

      return 1 - math::determinant(math::inner_square(A, W_pda)) / denominator;
    }

    Projector<T> optimize(
      const stats::Data<T> &      data,
      const stats::DataColumn<G> &groups,
      const std::set<G> &         unique_groups) const override {
      LOG_INFO << "Calculating PDA optimum projector for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
      LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables:" << std::endl;
      LOG_INFO << std::endl << data << std::endl;
      LOG_INFO << "Groups:" << std::endl;
      LOG_INFO << std::endl << groups << std::endl;

      stats::Data<T> B = stats::between_groups_sum_of_squares(data, groups, unique_groups);
      stats::Data<T> W = stats::within_groups_sum_of_squares(data, groups, unique_groups);

      LOG_INFO << "B:" << std::endl << B << std::endl;
      LOG_INFO << "W:" << std::endl << W << std::endl;

      stats::Data<T> W_diag = W.diagonal().asDiagonal();
      stats::Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
      stats::Data<T> WpB = W_pda + B;

      LOG_INFO << "W_pda:" << std::endl << W_pda << std::endl;
      LOG_INFO << "W_pda + B:" << std::endl << WpB << std::endl;

      stats::Data<T> WpBInvB = math::solve(WpB, B);
      stats::Data<T> truncatedWpBInvB = WpBInvB.unaryExpr(reinterpret_cast<T (*)(T)>(&math::truncate<T>));

      LOG_INFO << "(W_pda + B)^-1 * B:" << std::endl << WpBInvB << std::endl;
      LOG_INFO << "(W_pda + B)^-1 * B (truncated):" << std::endl << truncatedWpBInvB << std::endl;

      auto [eigen_val, eigen_vec] = math::eigen(truncatedWpBInvB);

      LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
      LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

      math::DVector<T> max_eigen_vec = eigen_vec.col(eigen_vec.cols() - 1);

      LOG_INFO << "Maximal eigenvector:" << std::endl << max_eigen_vec << std::endl;

      Projector<T> projector = pp::normalize(max_eigen_vec);

      LOG_INFO << "Projector:" << std::endl << projector << std::endl;
      return projector;
    }
  };

  template<typename T, typename G>
  std::unique_ptr<PPStrategy<T, G> > glda(const double lambda) {
    return std::make_unique<GLDAStrategy<T, G> >(lambda);
  }
}
