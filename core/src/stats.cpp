#include "stats.hpp"

namespace stats {
DMatrix<double> select_group(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned short          group
  ) {
  std::vector<unsigned short> index;

  for (unsigned short i = 0; i < groups.rows(); i++) {
    if (groups(i) == group) {
      index.push_back(i);
    }
  }

  if (index.size() == 0) {
    return DMatrix<double>(0, 0);
  }

  return data(index, Eigen::all);
}

DMatrix<double> between_groups_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count
  ) {
  DVector<double> global_mean = mean(data);
  DMatrix<double> result = DMatrix<double>::Zero(data.cols(), data.cols());

  for (unsigned short g = 0; g < group_count; g++) {
    DMatrix<double> group_data = select_group(data, groups, g);
    DVector<double> group_mean = mean(group_data);

    result += group_data.rows() * outer_square(group_mean - global_mean);
  }

  return result;
}

DMatrix<double> within_groups_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count
  ) {
  DMatrix<double> result = DMatrix<double>::Zero(data.cols(), data.cols());

  for (unsigned short g = 0; g < group_count; g++) {
    DMatrix<double> group_data = select_group(data, groups, g);
    DVector<double> group_mean = mean(group_data);
    DMatrix<double> centered_data = group_data.rowwise() - group_mean.transpose();

    for (unsigned int r = 0; r < centered_data.rows(); r++) {
      result += outer_square(centered_data.row(r));
    }
  }

  return result;
}
}
