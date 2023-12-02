#include <Eigen/Dense>
#include <vector>

namespace stats {
using namespace Eigen;

template<typename T = double>
using DMatrix = Eigen::Matrix<T, Dynamic, Dynamic>;

template<typename T = double>
using DVector = Eigen::Matrix<T, 1, Dynamic>;

DVector<double> mean(
  DMatrix<double> data);

DMatrix<double> outer_product(
  DVector<double> a,
  DVector<double> b);

DMatrix<double> outer_square(
  DVector<double> a);

double inner_product(
  DVector<double>  a,
  DVector<double>  b,
  DMatrix <double> weights);

double inner_square(
  DVector<double> a,
  DMatrix<double> weights);

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
