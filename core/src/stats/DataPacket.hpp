#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <set>
#include <vector>

namespace pptree::stats {
  struct DataPacket {
    const types::FeatureMatrix x;
    const types::Vector<types::Response>  y;
    const std::set<types::Response>  classes;

    DataPacket(
      const types::FeatureMatrix &           x,
      const types::Vector<types::Response> & y,
      const std::set<types::Response> &      classes) :
      x(x),
      y(y),
      classes(classes) {
    }

    DataPacket(
      const types::FeatureMatrix &           x,
      const types::Vector<types::Response> & y) :
      x(x),
      y(y),
      classes(unique(y)) {
    }

    DataPacket() :
      x(types::FeatureMatrix()),
      y(types::Vector<types::Response>()),
      classes(std::set<types::Response>()) {
    }
  };
}
