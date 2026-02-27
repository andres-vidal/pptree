#pragma once

#include "Types.hpp"
namespace models::pp {
  using Projector = types::FeatureVector;

  Projector normalize(const Projector &projector);
}
