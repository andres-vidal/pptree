#pragma once

#include "models/Projector.hpp"
#include "models/Strategy.hpp"
#include "utils/Types.hpp"

/**
 * @brief Split rule strategies for computing decision thresholds.
 *
 * Contains the abstract SRStrategy interface and concrete
 * implementations that determine the split threshold in projected
 * space.  The built-in SRMeanOfMeansStrategy uses the midpoint
 * of the two group means.
 *
 * New strategies must implement the pure virtual methods including
 * to_json() for serialization support.
 */
namespace ppforest2::sr {
  /**
   * @brief Abstract strategy for computing the split threshold.
   *
   * Given the data for two groups and a projection vector, determines
   * the threshold value that separates the groups in the projected
   * space.  Concrete subclasses implement different splitting rules
   * (e.g. mean of group means, median-based rules).
   *
   * Implementations must also provide to_json() for serialization.
   */
  struct SRStrategy : public Strategy<SRStrategy> {
    /**
     * @brief Compute the split threshold for two groups.
     *
     * @param group_1    Feature matrix for the first group (n1 × p).
     * @param group_2    Feature matrix for the second group (n2 × p).
     * @param projector  Projection vector (p).
     * @return           Scalar threshold in the projected space.
     */
    virtual types::Feature threshold(
      const types::FeatureMatrix& group_1,
      const types::FeatureMatrix& group_2,
      const pp::Projector&        projector) const = 0;

    /**
     * @brief Convenience operator: equivalent to threshold().
     */
    types::Feature operator()(
      const types::FeatureMatrix& group_1,
      const types::FeatureMatrix& group_2,
      const pp::Projector&        projector) const {
      return threshold(group_1, group_2, projector);
    }
  };
}
