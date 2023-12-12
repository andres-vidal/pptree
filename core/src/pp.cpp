#include "pp.hpp"


namespace pp {
DVector<double> lda_optimum_projector(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count) {
  DMatrix<double> W = within_groups_sum_of_squares(data, groups, group_count);
  DMatrix<double> B = between_groups_sum_of_squares(data, groups, group_count);

  auto [eigen_val, eigen_vec] = eigen(linear_algebra::inverse(W + B) * B);

  return eigen_vec(all, last);
}

double lda_index(
  DMatrix<double>         data,
  DMatrix<double>         projection_vector,
  DVector<unsigned short> groups,
  unsigned int            group_count) {
  DMatrix<double> A = projection_vector;

  DMatrix<double> W = within_groups_sum_of_squares(data, groups, group_count);
  DMatrix<double> B = between_groups_sum_of_squares(data, groups, group_count);

  double denominator = determinant(inner_square(A, W + B));

  if (denominator == 0) {
    return 0;
  }

  return 1 - determinant(inner_square(A, W)) / denominator;
}
}
