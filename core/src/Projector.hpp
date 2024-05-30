#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"
#include "DVector.hpp"

namespace pptree::pp {
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
}
