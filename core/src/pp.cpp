#include "pp.hpp"

using namespace pp;
using namespace linalg;
using namespace stats;
using namespace Eigen;
namespace pp {
Projector<double> lda_optimum_projector(
  Data<double>               data,
  DataColumn<unsigned short> groups,
  unsigned int               group_count) {
  DMatrix<double> W = within_groups_sum_of_squares(data, groups, group_count);
  DMatrix<double> B = between_groups_sum_of_squares(data, groups, group_count);

  auto [eigen_val, eigen_vec] = eigen(linalg::inverse(W + B) * B);

  return eigen_vec(all, last);
}

double lda_index(
  Data<double>               data,
  Projector<double>          projector,
  DataColumn<unsigned short> groups,
  unsigned int               group_count) {
  DMatrix<double> A = projector;

  DMatrix<double> W = within_groups_sum_of_squares(data, groups, group_count);
  DMatrix<double> B = between_groups_sum_of_squares(data, groups, group_count);

  double denominator = determinant(inner_square(A, W + B));

  if (denominator == 0) {
    return 0;
  }

  return 1 - determinant(inner_square(A, W)) / denominator;
}

DataColumn<double> project(
  Data<double>      data,
  Projector<double> projector) {
  return data * projector;
}
}
