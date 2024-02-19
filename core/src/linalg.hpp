#include <Eigen/Dense>

namespace linalg {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  DVector<long double> mean(
    const DMatrix<long double> &data);

  DMatrix<long double> outer_product(
    const DVector<long double> &a,
    const DVector<long double> &b);

  DMatrix<long double> outer_square(
    const DVector<long double> &a);

  long double inner_product(
    const DVector<long double> &a,
    const DVector<long double> &b,
    const DMatrix<long double> &weights);

  long double inner_product(
    const DVector<long double> &a,
    const DVector<long double> &b);

  long double inner_square(
    const DVector<long double> &a,
    const DMatrix<long double> &weights);

  DMatrix<long double> inner_product(
    const DMatrix<long double> &a,
    const DMatrix<long double> &b,
    const DMatrix<long double> &weights);

  DMatrix<long double> inner_product(
    const DMatrix<long double> &a,
    const DMatrix<long double> &b);

  DMatrix<long double> inner_square(
    const DMatrix<long double> &m,
    const DMatrix<long double> &weights);

  long double determinant(
    const DMatrix<long double> &m);

  DMatrix<long double> inverse(
    const DMatrix<long double> &m);

  std::tuple<DVector<long double>, DMatrix<long double> > eigen(
    const DMatrix<long double> &m);

  bool collinear(
    const DVector<long double> &a,
    const DVector<long double> &b);
}
