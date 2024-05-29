#pragma once

#include "Projector.hpp"

template<typename T>
using PPStrategyReturn = std::tuple<Projector<T>, Projection<T> >;
template<typename T, typename G>
using PPStrategy = std::function<PPStrategyReturn<T>(const Data<T>&, const DataColumn<G>&, const std::set<G>&)>;

template<typename T, typename G>
Projector<T> glda_optimum_projector(
  const Data<T> &      data,
  const DataColumn<G> &groups,
  const std::set<G> &  unique_groups,
  const double         lambda);


template<typename T, typename G>
T glda_index(
  const Data<T> &      data,
  const Projector<T> & projector,
  const DataColumn<G> &groups,
  const std::set<G> &  unique_groups,
  const double         lambda) {
  Data<T> A = projector;

  Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
  Data<T> W_diag = W.diagonal().asDiagonal();
  Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
  Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);
  Data<T> WpB = W_pda + B;

  T denominator = determinant(inner_square(A, WpB));

  if (denominator == 0) {
    return 0;
  }

  return 1 - determinant(inner_square(A, W_pda)) / denominator;
}

template<typename T, typename G>
PPStrategy<T, G> glda(
  const double lambda) {
  if (lambda == 0) {
    LOG_INFO << "Chosen Projection-Pursuit Strategy is LDA" << std::endl;
  } else {
    LOG_INFO << "Chosen Projection-Pursuit Strategy is PDA(lambda = " << lambda << ")" << std::endl;
  }

  return [lambda](const Data<T>& data, const DataColumn<G>& groups, const std::set<G>& unique_groups) -> PPStrategyReturn<T> {
           auto projector = glda_optimum_projector(data, groups, unique_groups, lambda);
           return PPStrategyReturn<T> { projector, project(data, projector) };
  };
}
