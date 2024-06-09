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
      const stats::Data<T>&       data,
      const stats::DataColumn<G>& groups,
      const std::set<G>&          unique_groups) const override;
  };


  template<typename T, typename G>
  std::unique_ptr<PPStrategy<T, G> > glda(const double lambda) {
    return std::make_unique<GLDAStrategy<T, G> >(lambda);
  }
}
