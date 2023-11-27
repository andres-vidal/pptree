#include "stat.hpp"

#include <vector>
#include <iostream>

DVector<double> mean(
  DMatrix<double> data
  ) {
  return data.colwise().mean();
}

DMatrix<double> select_group(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned short          group
  ) {
  std::vector<unsigned short> index;

  for (unsigned short i = 0; i < groups.cols(); i++) {
    if (groups(i) == group) {
      index.push_back(i);
    }
  }

  if (index.size() == 0) {
    return DMatrix<double>(0, 0);
  }

  return data(index, Eigen::all);
}

double inter_group_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count
  ) {
  DVector<double> global_mean = mean(data);

  double result = 0.0;

  for (unsigned short g = 0; g < group_count; g++) {
    DMatrix<double> group_data = select_group(data, groups, g);
    DVector<double> group_mean = mean(group_data);
    DVector<double> diff = group_mean - global_mean;
    result += group_data.rows() * diff.squaredNorm();
  }

  return result;
}

double intra_group_sum_of_squares(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count
  ) {
  double result = 0.0;

  for (unsigned short g = 0; g < group_count; g++) {
    DMatrix<double> group_data = select_group(data, groups, g);
    DVector<double> group_mean = mean(group_data);
    DMatrix<double> diff = group_data.rowwise() - group_mean;

    result += diff.squaredNorm();
  }

  return result;
}
