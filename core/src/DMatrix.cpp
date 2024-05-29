#include "DMatrix.hpp"
#include "Math.hpp"

template<typename T>
std::tuple<DVector<T>, DMatrix<T> > sort_eigen(
  const DVector<T> & values,
  const DMatrix<T> & vectors
  ) {
  DVector<int> idx = DVector<int>::Zero(values.size());

  for (int i = 0; i < values.size(); ++i) {
    idx[i] = i;
  }

  std::sort(idx.data(), idx.data() + idx.size(), [&values, &vectors](int idx1, int idx2)
  {
    long double value1_mod = fabs(values.row(idx1).value());
    long double value2_mod = fabs(values.row(idx2).value());

    if (is_approx(value1_mod, value2_mod)) {
      DVector<T> vector1 = vectors.col(idx1);
      DVector<T> vector2 = vectors.col(idx2);

      for (int i = 0; i < vector1.size(); ++i) {
        if (!is_module_approx(vector1[i], vector2[i]) ) {
          return vector1[i] < vector2[i];
        }
      }
    }

    return value1_mod < value2_mod;
  });


  return { values(idx), vectors(Eigen::all, idx) };
}

template<typename T>
std::tuple<DVector<T>, DMatrix<T> > eigen(const DMatrix<T> &m) {
  if (!m.isApprox(m.transpose())) {
    LOG_INFO << "Non-symmetric matrix detected, using general eigenvalue solver" << std::endl;

    Eigen::EigenSolver<DMatrix<T> > solver(m);
    DVector<T> values = solver.eigenvalues().real();
    DMatrix<T> vectors = solver.eigenvectors().real();

    return sort_eigen(values, vectors);
  }

  LOG_INFO << "Symmetric matrix detected, using self-adjoint eigenvalue solver" << std::endl;

  Eigen::SelfAdjointEigenSolver<DMatrix<T> > solver(m);
  DVector<T> values = solver.eigenvalues().real();
  DMatrix<T> vectors = solver.eigenvectors().real();

  return { values, vectors };
}

template std::tuple<DVector<long double>, DMatrix<long double> > eigen(const DMatrix<long double> &m);
