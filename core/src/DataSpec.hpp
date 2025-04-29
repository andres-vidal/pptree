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
      const std::set<G> &   classes)
      : x(x),
      y(y),
      classes(classes) {
    }

    DataSpec(
      const Data<T> &       x,
      const DataColumn<G> & y)
      : DataSpec(x, y, unique(y)) {
    }

    DataSpec(const DataSpec<T, G>& other)
      : x(other.x), y(other.y), classes(other.classes) {
    }
  };
}
