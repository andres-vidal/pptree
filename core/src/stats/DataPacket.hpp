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
    types::FeatureMatrix const x;
    /** @brief Outcome vector (n). */
    types::Vector<types::Outcome> const y;
    /** @brief Set of distinct group labels. */
    std::set<types::Outcome> const groups;
    /**
     * @brief Original group label names, indexed by integer code.
     *
     * When populated, group_names[i] is the original string label
     * that maps to integer code i.  Empty when data is not read
     * from CSV (e.g., simulated data).
     */
    std::vector<std::string> const group_names;
    /**
     * @brief Original feature column names from the CSV header.
     *
     * When populated, feature_names[j] is the header label for
     * column j of x.  Empty when data is simulated.
     */
    std::vector<std::string> const feature_names;

    DataPacket(
        types::FeatureMatrix const& x,
        types::Vector<types::Outcome> const& y,
        std::set<types::Outcome> const& groups,
        std::vector<std::string> const& group_names   = {},
        std::vector<std::string> const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups(groups)
        , group_names(group_names)
        , feature_names(feature_names) {}

    DataPacket(
        types::FeatureMatrix const& x,
        types::Vector<types::Outcome> const& y,
        std::vector<std::string> const& group_names   = {},
        std::vector<std::string> const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups(unique(y))
        , group_names(group_names)
        , feature_names(feature_names) {}

    DataPacket()
        : x(types::FeatureMatrix())
        , y(types::Vector<types::Outcome>())
        , groups(std::set<types::Outcome>()) {}
  };
}
