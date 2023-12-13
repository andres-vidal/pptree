#include "linear_algebra.hpp"

namespace stats {
linear_algebra::DMatrix<double> select_group(
  linear_algebra::DMatrix<double>         data,
  linear_algebra::DVector<unsigned short> groups,
  unsigned short                          group);

linear_algebra::DMatrix<double> between_groups_sum_of_squares(
  linear_algebra::DMatrix<double>         data,
  linear_algebra::DVector<unsigned short> groups,
  unsigned int                            group_count);

linear_algebra::DMatrix<double> within_groups_sum_of_squares(
  linear_algebra::DMatrix<double>         data,
  linear_algebra::DVector<unsigned short> groups,
  unsigned int                            group_count);
}
