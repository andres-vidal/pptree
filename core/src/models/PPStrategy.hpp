#pragma once

#include "models/Projector.hpp"
#include "models/Strategy.hpp"
#include "stats/GroupPartition.hpp"

/**
 * @brief Projection pursuit strategies.
 *
 * Contains the abstract PPStrategy interface and concrete
 * implementations (e.g. PPPDAStrategy) that define how to evaluate
 * and optimise a projection index for separating groups.
 *
 * New strategies must implement the pure virtual methods including
 * to_json() for serialization support.
 */
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
   *
   * Implementations must also provide to_json() for serialization.
   */
  struct PPStrategy : public Strategy<PPStrategy> {
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
