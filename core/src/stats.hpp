#include "linalg.hpp"

namespace stats {
linalg::DMatrix<double> select_group(
  linalg::DMatrix<double>         data,
  linalg::DVector<unsigned short> groups,
  unsigned short                  group);

linalg::DMatrix<double> between_groups_sum_of_squares(
  linalg::DMatrix<double>         data,
  linalg::DVector<unsigned short> groups,
  unsigned int                    group_count);

linalg::DMatrix<double> within_groups_sum_of_squares(
  linalg::DMatrix<double>         data,
  linalg::DVector<unsigned short> groups,
  unsigned int                    group_count);
}
