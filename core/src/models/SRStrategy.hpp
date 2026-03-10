#pragma once

#include "utils/Types.hpp"
#include "models/Projector.hpp"

#include <memory>

namespace pptree::sr {
  /**
   * @brief Abstract strategy for computing the split threshold.
   *
   * Given the data for two groups and a projection vector, determines
   * the threshold value that separates the groups in the projected
   * space.  Concrete subclasses implement different splitting rules
   * (e.g. mean of group means, median-based rules).
   */
  struct SRStrategy {
    using Ptr = std::unique_ptr<SRStrategy>;

    virtual ~SRStrategy()     = default;
    virtual Ptr clone() const = 0;

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
