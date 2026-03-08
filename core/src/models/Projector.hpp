#pragma once

#include "utils/Types.hpp"
namespace pptree::pp {
  using Projector = types::FeatureVector;

  Projector normalize(const Projector &projector);
}
