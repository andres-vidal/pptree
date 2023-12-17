#include "linalg.hpp"

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
  return inner_product(a, b, DMatrix<double>::Identity(a.size(), b.size()));
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

  if (!inverse.allFinite()) {
    std::stringstream message;
    message << "Matrix is not invertible:" << std::endl << m << std::endl;
    throw std::invalid_argument(message.str());
  }

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
    EigenSolver<DMatrix<double> > solver(m);
    DVector<double> values = solver.eigenvalues().real();
    DMatrix<double> vectors = solver.eigenvectors().real();

    return sort_eigen(values, vectors);
  }

  SelfAdjointEigenSolver<DMatrix<double> > solver(m);
  DVector<double> values = solver.eigenvalues().real();
  DMatrix<double> vectors = solver.eigenvectors().real();

  return std::make_tuple(values, vectors);
}
}
