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
