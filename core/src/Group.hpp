#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"

#include <set>

namespace pptree::stats {
  template<typename T, typename G>
  std::tuple<DataColumn<G>, std::set<int>, std::map<int, std::set<G> > > binary_regroup(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  unique_groups);
}
