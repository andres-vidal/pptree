#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"
#include "DVector.hpp"


template<typename T>
using Projector = DVector<T>;

template<typename T>
using Projection = DataColumn<T>;

template<typename T>
Projection<T> project(
  const Data<T> &     data,
  const Projector<T> &projector) {
  return data * projector;
}

template<typename T>
T project(
  const DataColumn<T> &data,
  const Projector<T> & projector) {
  return (data.transpose() * projector).value();
}
