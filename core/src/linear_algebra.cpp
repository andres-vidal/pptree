#include "linear_algebra.hpp"

namespace linear_algebra {
DVector<double> mean(
  DMatrix<double> data
  ) {
  return data.colwise().mean();
}

DMatrix<double> outer_product(
  DVector<double> a,
  DVector<double> b
  ) {
  return a.transpose() * b;
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
  return a * weights * b.transpose();
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
}
