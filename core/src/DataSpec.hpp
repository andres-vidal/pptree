#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"

#include <set>
#include <vector>

namespace pptree::stats {
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
      : x(x),
      y(y),
      classes(unique(y)) {
    }

    virtual std::tuple<Data<T>, DataColumn<G>, std::set<G> > unwrap() const {
      return std::make_tuple(x, y, classes);
    }
  };

  template<typename T, typename G>
  DataSpec<T, G> center(const DataSpec<T, G> &data) {
    return DataSpec<T, G>(center(data.x), data.y, data.classes);
  }

  template<typename T, typename G>
  DataSpec<T, G> descale(const DataSpec<T, G> &data) {
    return DataSpec<T, G>(descale(data.x), data.y, data.classes);
  }
}
