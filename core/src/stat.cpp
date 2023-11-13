#include "stat.hpp"

DVector<double> mean(
  DMatrix<double> data
  ) {
  return data.colwise().mean();
}
