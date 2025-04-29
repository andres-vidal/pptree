#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"
#include "Uniform.hpp"

#include <map>
#include <set>
#include <random>

namespace models::stats
{
  template <typename T>
  using Data = math::DMatrix<T>;

  template <typename T>
  using DataView = Eigen::Block<const Data<T> >;

  template <typename T>
  Data<T> shuffle_column(
    const Data<T> &data,
    const int      column) {
    Data<T> shuffled = data;

    Uniform unif(0, data.rows() - 1);

    std::vector<int> indices(data.rows());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = data.rows() - 1; i > 0; i--) {
      int j = unif();
      std::swap(indices[i], indices[j]);
    }

    for (int i = 0; i < data.rows(); i++) {
      shuffled(i, column) = data(indices[i], column);
    }

    return shuffled;
  }
}
