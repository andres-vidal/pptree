#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <set>
#include <string>
#include <vector>

namespace ppforest2::stats {
  /**
   * @brief Bundled dataset: features, responses, and group labels.
   *
   * Convenience struct that groups a feature matrix, a response vector,
   * and the set of unique group labels.  Used primarily for passing
   * data through the training pipeline.
   */
  struct DataPacket {
    /** @brief Feature matrix (n × p). */
    const types::FeatureMatrix x;
    /** @brief Response vector (n). */
    const types::Vector<types::Response>  y;
    /** @brief Set of distinct group labels. */
    const std::set<types::Response>  groups;
    /**
     * @brief Original group label names, indexed by integer code.
     *
     * When populated, group_names[i] is the original string label
     * that maps to integer code i.  Empty when data is not read
     * from CSV (e.g., simulated data).
     */
    const std::vector<std::string> group_names;

    DataPacket(
      const types::FeatureMatrix &           x,
      const types::Vector<types::Response> & y,
      const std::set<types::Response> &      groups,
      const std::vector<std::string> &       group_names = {}) :
      x(x),
      y(y),
      groups(groups),
      group_names(group_names) {
    }

    DataPacket(
      const types::FeatureMatrix &           x,
      const types::Vector<types::Response> & y,
      const std::vector<std::string> &       group_names = {}) :
      x(x),
      y(y),
      groups(unique(y)),
      group_names(group_names) {
    }

    DataPacket() :
      x(types::FeatureMatrix()),
      y(types::Vector<types::Response>()),
      groups(std::set<types::Response>()) {
    }
  };
}
