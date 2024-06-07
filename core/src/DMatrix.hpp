#pragma once

#include "Logger.hpp"
#include "DVector.hpp"

#include <Eigen/Dense>

namespace models::math {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template<typename T>
  DMatrix<T> inner_product(
    const DMatrix<T> &a,
    const DMatrix<T> &b,
    const DMatrix<T> &weights
    ) {
    return (a.transpose() * weights * b);
  }

  template<typename T>
  DMatrix<T> inner_product(
    const DMatrix<T> &a,
    const DMatrix<T> &b
    ) {
    DMatrix<T> identity = DMatrix<T>::Identity(a.rows(), b.cols());
    return inner_product(a, b, identity);
  }

  template<typename T>
  DMatrix<T> inner_square(
    const DMatrix<T> &m,
    const DMatrix<T> &weights
    ) {
    return inner_product(m, m, weights);
  }

  template<typename T>
  long double determinant(
    const DMatrix<T> &m
    ) {
    return m.determinant();
  }

  template<typename T>
  DMatrix<T> inverse(const DMatrix<T> &m) {
    Eigen::FullPivLU<DMatrix<T> > lu(m);

    assert(lu.isInvertible() && "Given matrix is not invertible");
    return lu.inverse();
  }

  template<typename T>
  DMatrix<T> solve(
    const DMatrix<T> &l,
    const DMatrix<T> &r
    ) {
    Eigen::FullPivLU<DMatrix<T> > lu(l);

    assert(lu.isInvertible() && "Given matrix is not invertible");
    return lu.solve(r);
  }

  template<typename T>
  bool collinear(
    const DMatrix<T> &a,
    const DMatrix<T> &b) {
    for (int i = 0; i < a.cols(); i++) {
      DVector<T> a_col = a.col(i);
      DVector<T> b_col = b.col(i);

      if (!collinear(a_col, b_col)) {
        return false;
      }
    }

    return true;
  }

  template<typename T>
  std::tuple<DVector<T>, DMatrix<T> > eigen(const DMatrix<T> &m);


  template<typename T>
  T trace(const DMatrix<T> &m) {
    return m.trace();
  }

  template<typename T>
  T sum(const DMatrix<T> &m) {
    return m.rowwise().sum().colwise().sum().value();
  }
}
