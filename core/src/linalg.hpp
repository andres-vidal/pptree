#include <Eigen/Dense>

namespace linalg {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  DVector<double> mean(
    const DMatrix<double> &data);

  DMatrix<double> outer_product(
    const DVector<double> &a,
    const DVector<double> &b);

  DMatrix<double> outer_square(
    const DVector<double> &a);

  double inner_product(
    const DVector<double> &a,
    const DVector<double> &b,
    const DMatrix<double> &weights);

  double inner_product(
    const DVector<double> &a,
    const DVector<double> &b);

  double inner_square(
    const DVector<double> &a,
    const DMatrix<double> &weights);

  DMatrix<double> inner_product(
    const DMatrix<double> &a,
    const DMatrix<double> &b,
    const DMatrix<double> &weights);

  DMatrix<double> inner_product(
    const DMatrix<double> &a,
    const DMatrix<double> &b);

  DMatrix<double> inner_square(
    const DMatrix<double> &m,
    const DMatrix<double> &weights);

  double determinant(
    const DMatrix<double> &m);

  DMatrix<double> inverse(
    const DMatrix<double> &m);

  std::tuple<DVector<double>, DMatrix<double> > eigen(
    const DMatrix<double> &m);

  bool collinear(
    const DVector<double> &a,
    const DVector<double> &b);
}
