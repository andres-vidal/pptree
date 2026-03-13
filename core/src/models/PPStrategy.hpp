#pragma once

#include "models/Projector.hpp"
#include "stats/GroupPartition.hpp"
#include <set>
#include <vector>
#include <memory>

namespace ppforest2::pp {
  /**
   * @brief Result of a projection pursuit optimization step.
   */
  struct PPResult {
    /** @brief Optimized projection vector. */
    Projector projector;
    /** @brief Projection pursuit index value achieved. */
    types::Feature index_value = 0;
  };

  /**
   * @brief Abstract strategy for projection pursuit optimization.
   *
   * Defines how to evaluate a projection (via an index function) and
   * how to find the optimal projection for a given dataset and
   * group partition.
   */
  struct PPStrategy {
    using Ptr = std::unique_ptr<PPStrategy>;

    virtual ~PPStrategy()     = default;
    virtual Ptr clone() const = 0;

    /**
     * @brief Evaluate the projection pursuit index for a given projector.
     *
     * @param x           Feature matrix (n × p).
     * @param group_spec  Group partition.
     * @param projector   Projection vector (p).
     * @return            Index value (higher is better separation).
     */
    virtual types::Feature index(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec,
      const Projector&             projector) const = 0;

    /**
     * @brief Find the optimal projection for the data.
     *
     * @param x           Feature matrix (n × p).
     * @param group_spec  Group partition.
     * @return            Optimized projector and its index value.
     */
    virtual PPResult optimize(
      const types::FeatureMatrix&  x,
      const stats::GroupPartition& group_spec) const = 0;

    /**
     * @brief Convenience operator: optimize and return the projector only.
     */
    Projector operator()(const types::FeatureMatrix &x, const stats::GroupPartition& group_spec) const {
      return optimize(x, group_spec).projector;
    }
  };
}
