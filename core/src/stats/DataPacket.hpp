#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <set>
#include <string>
#include <vector>

namespace ppforest2::stats {
  /**
   * @brief Bundled dataset: features, response, and group labels.
   *
   * Convenience struct that groups a feature matrix and a response vector
   * with dataset-level metadata (unique group labels, column names).  Used
   * primarily for passing data through the training pipeline.
   *
   * `y` is carried as `OutcomeVector` (float) for both modes — integer
   * class labels for classification and the continuous response for
   * regression. Sites that require integer group identity (GroupPartition
   * construction, ConfusionMatrix indexing, stratified sampling, factor
   * label lookup) cast locally via `types::y.cast<GroupId>()`.
   */
  struct DataPacket {
    // Members are non-const: `DataPacket` is a transient training-pipeline
    // carrier. Regression strategies (ByCutpoint) permute rows of `x` and
    // `y` in place during training. CLI and R bindings discard the packet
    // after the training call, so mutation is invisible to end users.

    /** @brief Feature matrix (n × p). */
    types::FeatureMatrix x;
    /** @brief Response vector (n) — integer labels (classification) or continuous response (regression). */
    types::OutcomeVector y;
    /** @brief Set of distinct group labels (classification only; empty for regression). */
    std::set<types::GroupId> groups;
    /**
     * @brief Original group label names, indexed by integer code.
     *
     * When populated, group_names[i] is the original string label
     * that maps to integer code i.  Empty when data is not read
     * from CSV (e.g., simulated data) or for regression.
     */
    std::vector<std::string> group_names;
    /**
     * @brief Original feature column names from the CSV header.
     *
     * When populated, feature_names[j] is the header label for
     * column j of x.  Empty when data is simulated.
     */
    std::vector<std::string> feature_names;

    DataPacket(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        std::set<types::GroupId> const& groups,
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
        types::OutcomeVector const& y,
        std::vector<std::string> const& group_names   = {},
        std::vector<std::string> const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups(unique(y.cast<types::GroupId>()))
        , group_names(group_names)
        , feature_names(feature_names) {}

    /**
     * @brief Construct without a groups set (regression).
     *
     * Used by regression data readers where `y` is the continuous
     * response and there are no discrete group labels to enumerate.
     */
    struct NoGroups {};
    DataPacket(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        NoGroups,
        std::vector<std::string> const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups{}
        , feature_names(feature_names) {}

    DataPacket()
        : x(types::FeatureMatrix())
        , y(types::OutcomeVector())
        , groups(std::set<types::GroupId>()) {}
  };
}
