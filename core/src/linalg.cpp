#include "linalg.hpp"
#include "pptreeio.hpp"

using namespace Eigen;
using namespace linalg;

namespace linalg {
  DVector<long double> mean(
    const DMatrix<long double> &data
    ) {
    return data.colwise().mean();
  }

  DMatrix<long double> outer_product(
    const DVector<long double> &a,
    const DVector<long double> &b
    ) {
    return a * b.transpose();
  }

  DMatrix<long double> outer_square(
    const DVector<long double> &a
    ) {
    return outer_product(a, a);
  }

  long double inner_product(
    const DVector<long double> &a,
    const DVector<long double> &b,
    const DMatrix<long double> &weights) {
    return a.transpose() * weights * b;
  }

  long double inner_product(
    const DVector<long double> &a,
    const DVector<long double> &b
    ) {
    return inner_product(a, b, DMatrix<long double>::Identity(a.size(), b.size()));
  }

  long double inner_square(
    const DVector<long double> &a,
    const DMatrix<long double> &weights
    ) {
    return inner_product(a, a, weights);
  }

  DMatrix<long double> inner_product(
    const DMatrix<long double> &a,
    const DMatrix<long double> &b,
    const DMatrix<long double> &weights) {
    return (a.transpose() * weights * b);
  }

  DMatrix<long double> inner_product(
    const DMatrix<long double> &a,
    const DMatrix<long double> &b
    ) {
    return inner_product(a, b, DMatrix<long double>::Identity(a.rows(), b.cols()));
  }

  DMatrix<long double> inner_square(
    const DMatrix<long double> &m,
    const DMatrix<long double> &weights
    ) {
    return inner_product(m, m, weights);
  }

  long double determinant(
    const DMatrix<long double> &m
    ) {
    return m.determinant();
  }

  DMatrix<long double> inverse(
    const DMatrix<long double> &m
    ) {
    DMatrix<long double> inverse = m.inverse();
    assert(inverse.allFinite() && "Given matrix is not invertible");
    return inverse;
  }

  std::tuple<DVector<long double>, DMatrix<long double> > sort_eigen(
    const DVector<long double> &values,
    const DMatrix<long double> &vectors
    ) {
    DVector<int> idx = DVector<int>::Zero(values.size());

    for (int i = 0; i < values.size(); ++i) {
      idx[i] = i;
    }

    std::sort(idx.data(), idx.data() + idx.size(), [&values](long double a, long double b)
    {
      return abs(values.row(a).value()) < abs(values.row(b).value());
    });

    return std::make_tuple(values(idx), vectors(all, idx));
  }

  std::tuple<DVector<long double>, DMatrix<long double> > eigen(
    const DMatrix<long double> &m
    ) {
    if (!m.isApprox(m.transpose())) {
      LOG_INFO << "Non-symmetric matrix detected, using general eigenvalue solver" << std::endl;

      EigenSolver<DMatrix<long double> > solver(m);
      DVector<long double> values = solver.eigenvalues().real();
      DMatrix<long double> vectors = solver.eigenvectors().real();

      return sort_eigen(values, vectors);
    }

    LOG_INFO << "Symmetric matrix detected, using self-adjoint eigenvalue solver" << std::endl;

    SelfAdjointEigenSolver<DMatrix<long double> > solver(m);
    DVector<long double> values = solver.eigenvalues().real();
    DMatrix<long double> vectors = solver.eigenvectors().real();

    return std::make_tuple(values, vectors);
  }

  bool collinear(
    const DVector<long double> &a,
    const DVector<long double> &b) {
    long double tolerance = 0.0001;

    return fabs(inner_product(a, b) / (a.norm() * b.norm()) - 1.0) < tolerance;
  }
}
