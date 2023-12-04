#include "linear_algebra.hpp"

using namespace linear_algebra;

namespace stats {
DMatrix<double> select_group(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned short          group);

DMatrix<double> between_groups_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count);

DMatrix<double> within_groups_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count);
}
