#include <Eigen/Dense>

using namespace Eigen;

template<typename T = double>
using DMatrix = Eigen::Matrix<T, Dynamic, Dynamic>;

template<typename T = double>
using DVector = Eigen::Matrix<T, 1, Dynamic>;

DVector<double> mean(
  DMatrix<double> data);

DMatrix<double> select_group(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned short          group);
