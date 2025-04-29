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

    virtual std::tuple<Data<T>, DataColumn<G>, std::set<G> > unwrap() const {
      return std::make_tuple(x, y, classes);
    }

    DataSpec<T, G> standardize() const {
      // Data<T> centered_x = x.colwise() - x.rowwise().mean();
      // DataColumn<T> sd_x = (centered_x.array().square().rowwise().sum() / (x.rows() - 1)).sqrt();
      // Data<T> descaled_x = (centered_x.array().rowwise() / sd_x.transpose().array()).matrix();

      return center(descale(*this));
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
