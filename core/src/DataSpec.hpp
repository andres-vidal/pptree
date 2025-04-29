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

    DataSpec<T, G> standardize() const {
      Data<T> centered_x = x.rowwise() - x.colwise().mean();
      DataColumn<T> sd_x = (centered_x.array().square().colwise().sum() / (x.rows() - 1)).sqrt();
      Data<T> descaled_x = centered_x.array().rowwise() / sd_x.transpose().array();

      return DataSpec<T, G>(descaled_x, y, classes);
    }
  };
}
