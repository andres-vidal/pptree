#pragma once

#include "DVector.hpp"

#include <set>

namespace pptree::stats {
  template<typename T>
  using DataColumn = math::DVector<T>;

  template<typename N>
  std::set<N> unique(const DataColumn<N> &column) {
    std::set<N> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> &   data,
    const std::vector<int> &indices) {
    DataColumn<T> result(indices.size());

    for (int i = 0; i < indices.size(); i++) {
      result(i) = data(indices[i]);
    }

    return result;
  }

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> & data,
    const std::set<int> & indices) {
    return select_rows(data, std::vector<int>(indices.begin(), indices.end()));
  }

  template<typename G>
  std::vector<G> select_group(
    const DataColumn<G> &groups,
    const G &            group) {
    std::vector<G> indices;

    for (int i = 0; i < groups.rows(); i++) {
      if (groups(i) == group) {
        indices.push_back(i);
      }
    }

    return indices;
  }

  template<typename T, typename G>
  DataColumn<T> select_group(
    const DataColumn<T> &data,
    const DataColumn<G> &groups,
    const G &            group) {
    std::vector<G> indices = select_group(groups, group);

    DataColumn<T> result(indices.size());

    for (int i = 0; i < indices.size(); i++) {
      result(i) = data(indices[i]);
    }

    return result;
  }

  template<typename T>
  DataColumn<T> expand(
    const DataColumn<T> &   data,
    const std::vector<int> &mask) {
    DataColumn<T> expanded = DataColumn<T>::Zero(mask.size());

    int j = 0;

    for (int i = 0; i < mask.size(); i++) {
      if (mask[i] == 1) {
        expanded.row(i) = data.row(j);
        j++;
      }
    }

    return expanded;
  }

  template<typename T>
  T mean(const DataColumn<T> &data) {
    return data.mean();
  }

  template<typename T>
  DataColumn<T> center(const DataColumn<T> &data) {
    return data.array() - mean(data);
  }

  template<typename T>
  T sd(const DataColumn<T> &data) {
    return sqrt((math::inner_square(center(data))) / (data.rows() - 1));
  }

  template<typename T>
  DataColumn<T> descale(const DataColumn<T> &data) {
    T scaling_factor = sd(data);

    if (scaling_factor == 0) {
      scaling_factor = 1;
    }

    return data.array() / scaling_factor;
  }
}
