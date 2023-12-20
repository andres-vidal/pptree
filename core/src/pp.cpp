#include "pp.hpp"

using namespace pp;
using namespace stats;
using namespace linalg;

namespace pp {
template<typename T, typename G>
Projector<T> lda_optimum_projector(
  Data<T>       data,
  DataColumn<G> groups,
  std::set<G>   unique_groups) {
  Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
  Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

  auto [eigen_val, eigen_vec] = linalg::eigen(linalg::inverse(W + B) * B);

  return eigen_vec(Eigen::all, Eigen::last);
}

template Projector<double> lda_optimum_projector<double, int>(
  Data<double>    data,
  DataColumn<int> groups,
  std::set<int>   unique_groups);

template<typename T, typename G>
T lda_index(
  Data<T>       data,
  Projector<T>  projector,
  DataColumn<G> groups,
  std::set<G>   unique_groups) {
  Data<T> A = projector;

  Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
  Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

  T denominator = linalg::determinant(linalg::inner_square(A, W + B));

  if (denominator == 0) {
    return 0;
  }

  return 1 - determinant(inner_square(A, W)) / denominator;
}

template double lda_index<double, int>(
  Data<double>      data,
  Projector<double> projector,
  DataColumn<int>   groups,
  std::set<int>     unique_groups);

template<typename T>
Projection<T> project(
  Data<T>      data,
  Projector<T> projector) {
  return data * projector;
}

template Projection<double> project<double>(
  Data<double>      data,
  Projector<double> projector);
}
