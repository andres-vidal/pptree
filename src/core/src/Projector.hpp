#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"
#include "DVector.hpp"

namespace models::pp {
  template<typename T>
  using Projector = math::DVector<T>;

  template<typename T>
  using Projection = stats::DataColumn<T>;

  template<typename T>
  Projector<T> normalize(
    const Projector<T> &projector) {
    Projector<T> truncated = (projector.array().abs() < 1e-15).select(0, projector.array());

    const int size = truncated.size();
    const T *data  = truncated.data();

    int i = 0;

    while (i < size && math::is_approx(data[i], T(0), T(0.001)))
      ++i;

    if (i == size) {
      return truncated;
    }

    T sign = (data[i] < T(0)) ? T(-1) : T(1);

    return sign * truncated;
  }
}
