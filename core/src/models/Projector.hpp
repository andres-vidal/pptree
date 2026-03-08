#pragma once

#include "utils/Types.hpp"
namespace pptree::pp {
  /** @brief Column vector of projection coefficients (one per variable). */
  using Projector = types::FeatureVector;

  /**
   * @brief Normalize a projector to unit length.
   *
   * @param projector  Input projector (p).
   * @return           Normalized projector with L2 norm = 1.
   */
  Projector normalize(const Projector &projector);
}
