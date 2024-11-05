#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"
#include "Uniform.hpp"

#include <map>
#include <set>
#include <random>

namespace models::stats
{
  template <typename T>
  using Data = math::DMatrix<T>;

  template <typename T>
  Data<T> select_rows(
    const Data<T> &         data,
    const std::vector<int> &indices) {
    Data<T> result(indices.size(), data.cols());

    for (std::size_t i = 0; i < indices.size(); i++) {
      result.row(i) = data.row(indices[i]);
    }

    return result;
  }

  template <typename T>
  Data<T> select_rows(
    const Data<T> &      data,
    const std::set<int> &indices) {
    return select_rows(data, std::vector<int>(indices.begin(), indices.end()));
  }

  template <typename T, typename G>
  Data<T> select_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group) {
    std::vector<G> indices = select_group(groups, group);

    if (indices.size() == 0) {
      return Data<T>(0, 0);
    }

    return data(indices, Eigen::all);
  }

  template <typename T>
  std::tuple<std::vector<int>, std::vector<int> > mask_null_columns(const Data<T> &data) {
    std::vector<int> mask(data.cols());
    std::vector<int> index;

    for (int i = 0; i < data.cols(); i++) {
      if (data.col(i).minCoeff() == 0 && data.col(i).maxCoeff() == 0) {
        mask[i] = 0;
      } else {
        mask[i] = 1;
        index.push_back(i);
      }
    }

    return { mask, index };
  }

  template <typename T>
  DataColumn<T> mean(const Data<T> &data) {
    return data.colwise().mean();
  }

  template <typename T>
  Data<T> center(const Data<T> &data) {
    return data.rowwise() - mean(data).transpose();
  }

  template <typename T>
  Data<T> covariance(const Data<T> &data) {
    Data<T> centered = center(data);

    return (centered.transpose() * centered) / (data.rows() - 1);
  }

  template <typename T>
  DataColumn<T> sd(const Data<T> &data) {
    return covariance(data).diagonal().array().sqrt();
  }

  template <typename T>
  Data<T> descale(const Data<T> &data) {
    DataColumn<T> scaling_factor = sd(data);

    for (int i = 0; i < scaling_factor.rows(); i++) {
      if (scaling_factor(i) == 0) {
        scaling_factor(i) = 1;
      }
    }

    return data.array().rowwise() / scaling_factor.transpose().array();
  }

  template <typename T, typename G>
  Data<T> between_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups) {
    DataColumn<T> global_mean = mean(data);
    Data<T> result = Data<T>::Zero(data.cols(), data.cols());

    for (const G &group : unique_groups) {
      Data<T> group_data = select_group(data, groups, group);
      DataColumn<T> group_mean = mean(group_data);
      DataColumn<T> centered_mean = group_mean - global_mean;

      result += group_data.rows() * math::outer_square(centered_mean);
    }

    return result;
  }

  template <typename T, typename G>
  Data<T> within_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups) {
    Data<T> result = Data<T>::Zero(data.cols(), data.cols());

    for (const G &group : unique_groups) {
      Data<T> group_data = select_group(data, groups, group);
      Data<T> centered_group_data = center(group_data);

      result += math::inner_square(centered_group_data);
    }

    return result;
  }

  template <typename T>
  Data<T> shuffle_column(
    const Data<T> &data,
    const int      column) {
    Data<T> shuffled = data;

    Uniform unif(0, data.rows() - 1);

    std::vector<int> indices(data.rows());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = data.rows() - 1; i > 0; i--) {
      int j = unif();
      std::swap(indices[i], indices[j]);
    }

    for (int i = 0; i < data.rows(); i++) {
      shuffled(i, column) = data(indices[i], column);
    }

    return shuffled;
  }
}
