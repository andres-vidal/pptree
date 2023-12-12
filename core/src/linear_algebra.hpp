#include <Eigen/Dense>

namespace linear_algebra {
using namespace Eigen;

template<typename T = double>
using DMatrix = Matrix<T, Dynamic, Dynamic>;

template<typename T = double>
using DVector = Matrix<T, Dynamic, 1>;

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

DMatrix<double> inner_product(
  DMatrix<double> a,
  DMatrix<double> b,
  DMatrix<double> weights);

DMatrix<double> inner_square(
  DMatrix<double> m,
  DMatrix<double> weights);

double determinant(
  DMatrix<double> m);

DMatrix<double> inverse(
  DMatrix<double> m);

std::tuple<DVector<double>, DMatrix<double> > eigen(
  DMatrix<double> m);
}
