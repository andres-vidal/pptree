#include "linalg.hpp"
#include "pptreeio.hpp"

using namespace Eigen;
using namespace linalg;

namespace linalg {
  DVector<double> mean(
    const DMatrix<double> &data
    ) {
    return data.colwise().mean();
  }

  DMatrix<double> outer_product(
    const DVector<double> &a,
    const DVector<double> &b
    ) {
    return a * b.transpose();
  }

  DMatrix<double> outer_square(
    const DVector<double> &a
    ) {
    return outer_product(a, a);
  }

  double inner_product(
    const DVector<double> &a,
    const DVector<double> &b,
    const DMatrix<double> &weights) {
    return a.transpose() * weights * b;
  }

  double inner_product(
    const DVector<double> &a,
    const DVector<double> &b
    ) {
    return inner_product(a, b, DMatrix<double>::Identity(a.size(), b.size()));
  }

  double inner_square(
    const DVector<double> &a,
    const DMatrix<double> &weights
    ) {
    return inner_product(a, a, weights);
  }

  DMatrix<double> inner_product(
    const DMatrix<double> &a,
    const DMatrix<double> &b,
    const DMatrix<double> &weights) {
    return (a.transpose() * weights * b);
  }

  DMatrix<double> inner_product(
    const DMatrix<double> &a,
    const DMatrix<double> &b
    ) {
    return inner_product(a, b, DMatrix<double>::Identity(a.rows(), b.cols()));
  }

  DMatrix<double> inner_square(
    const DMatrix<double> &m,
    const DMatrix<double> &weights
    ) {
    return inner_product(m, m, weights);
  }

  double determinant(
    const DMatrix<double> &m
    ) {
    return m.determinant();
  }

  DMatrix<double> inverse(
    const DMatrix<double> &m
    ) {
    DMatrix<double> inverse = m.inverse();
    assert(inverse.allFinite() && "Given matrix is not invertible");
    return inverse;
  }

  std::tuple<DVector<double>, DMatrix<double> > sort_eigen(
    const DVector<double> &values,
    const DMatrix<double> &vectors
    ) {
    DVector<int> idx = DVector<int>::Zero(values.size());

    for (int i = 0; i < values.size(); ++i) {
      idx[i] = i;
    }

    std::sort(idx.data(), idx.data() + idx.size(), [&values](double a, double b)
    {
      return abs(values.row(a).value()) < abs(values.row(b).value());
    });

    return std::make_tuple(values(idx), vectors(all, idx));
  }

  std::tuple<DVector<double>, DMatrix<double> > eigen(
    const DMatrix<double> &m
    ) {
    if (!m.isApprox(m.transpose())) {
      LOG_INFO << "Non-symmetric matrix detected, using general eigenvalue solver" << std::endl;

      EigenSolver<DMatrix<double> > solver(m);
      DVector<double> values = solver.eigenvalues().real();
      DMatrix<double> vectors = solver.eigenvectors().real();

      return sort_eigen(values, vectors);
    }

    LOG_INFO << "Symmetric matrix detected, using self-adjoint eigenvalue solver" << std::endl;

    SelfAdjointEigenSolver<DMatrix<double> > solver(m);
    DVector<double> values = solver.eigenvalues().real();
    DMatrix<double> vectors = solver.eigenvectors().real();

    return std::make_tuple(values, vectors);
  }

  bool collinear(
    const DVector<double> &a,
    const DVector<double> &b) {
    double tolerance = 0.0001;
    return abs(inner_product(a, b) / (a.norm() * b.norm()) - 1.0) < tolerance;
  }
}
