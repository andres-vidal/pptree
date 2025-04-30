#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"

#include <set>
#include <vector>

namespace models::stats {
  template<typename T, typename G>
  struct DataSpec {
    const Data<T>  x;
    const DataColumn<G>  y;
    const std::set<G>  classes;

    DataSpec(
      const Data<T> &       x,
      const DataColumn<G> & y,
      const std::set<G> &   classes) :
      x(x),
      y(y),
      classes(classes) {
    }

    DataSpec(
      const Data<T> &       x,
      const DataColumn<G> & y) :
      x(x),
      y(y),
      classes(unique(y)) {
    }

    DataSpec() :
      x(Data<T>()),
      y(DataColumn<G>()),
      classes(std::set<G>()) {
    }
  };
}
