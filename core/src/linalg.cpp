#include "linalg.hpp"
#include "pptreeio.hpp"

using namespace Eigen;
using namespace linalg;

namespace linalg {
  DVector<double> mean(
    DMatrix<double> data
    ) {
    return data.colwise().mean();
  }

  DMatrix<double> outer_product(
    DVector<double> a,
    DVector<double> b
    ) {
    return a * b.transpose();
  }

  DMatrix<double> outer_square(
    DVector<double> a
    ) {
    return outer_product(a, a);
  }

  double inner_product(
    DVector<double>  a,
    DVector<double>  b,
    DMatrix <double> weights) {
    return a.transpose() * weights * b;
  }

  double inner_product(
    DVector<double> a,
    DVector<double> b
    ) {
    return inner_product(a, b, DMatrix<double>::Identity(a.size(), b.size()));
  }

  double inner_square(
    DVector<double> a,
    DMatrix<double> weights
    ) {
    return inner_product(a, a, weights);
  }

  DMatrix<double> inner_product(
    DMatrix<double> a,
    DMatrix<double> b,
    DMatrix<double> weights) {
    return (a.transpose() * weights * b);
  }

  DMatrix<double> inner_product(
    DMatrix<double> a,
    DMatrix<double> b
    ) {
    return inner_product(a, b, DMatrix<double>::Identity(a.rows(), b.cols()));
  }

  DMatrix<double> inner_square(
    DMatrix<double> m,
    DMatrix<double> weights
    ) {
    return inner_product(m, m, weights);
  }

  double determinant(
    DMatrix<double> m
    ) {
    return m.determinant();
  }

  DMatrix<double> inverse(
    DMatrix<double> m
    ) {
    DMatrix<double> inverse = m.inverse();
    assert(inverse.allFinite() && "Given matrix is not invertible");
    return inverse;
  }

  std::tuple<DVector<double>, DMatrix<double> > sort_eigen(
    DVector<double> values,
    DMatrix<double> vectors
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
    DMatrix<double> m
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

  bool collinear(DVector<double> a, DVector<double> b) {
    double tolerance = 0.0001;
    return abs(inner_product(a, b) / (a.norm() * b.norm()) - 1.0) < tolerance;
  }
}
