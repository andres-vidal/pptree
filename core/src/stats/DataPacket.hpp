#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <set>
#include <vector>

namespace pptree::stats {
  /**
   * @brief Bundled dataset: features, responses, and class labels.
   *
   * Convenience struct that groups a feature matrix, a response vector,
   * and the set of unique class labels.  Used primarily for passing
   * data through the training pipeline.
   */
  struct DataPacket {
    /** @brief Feature matrix (n × p). */
    const types::FeatureMatrix x;
    /** @brief Response vector (n). */
    const types::Vector<types::Response>  y;
    /** @brief Set of distinct class labels. */
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
