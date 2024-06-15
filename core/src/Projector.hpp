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
  Projection<T> project(
    const stats::Data<T> & data,
    const Projector<T> &   projector) {
    return data * projector;
  }

  template<typename T>
  T project(
    const stats::DataColumn<T> &data,
    const Projector<T> &        projector) {
    return (data.transpose() * projector).value();
  }

  template<typename T>
  Projector<T> normalize(
    const Projector<T> &projector) {
    Projector<T> truncated = projector.unaryExpr(reinterpret_cast<T (*)(T)>(&math::truncate<T>));

    // Fetch the index of the first non-zero component
    int i = 0;

    while (i < truncated.size() && math::is_approx(truncated(i), 0))
      i++;

    // Guarantee the first non-zero component is positive
    return (truncated(i) < 0 ? -1 : 1) * truncated;
  }

  template<typename T>
  Projector<T> expand(
    const Projector<T> &    projector,
    const std::vector<int> &mask) {
    return stats::expand(projector, mask);
  }
}
