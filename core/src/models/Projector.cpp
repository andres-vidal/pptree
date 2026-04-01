#include "models/Projector.hpp"
#include "utils/Math.hpp"

using namespace ppforest2::types;

namespace ppforest2::pp {
  Projector normalize(Projector const& projector) {
    Projector truncated = (projector.array().abs() < 1e-15).select(0, projector.array());

    int const size      = truncated.size();
    Feature const* data = truncated.data();

    int i = 0;

    while (i < size && math::is_approx(data[i], Feature(0), Feature(0.001)))
      ++i;

    if (i == size) {
      return truncated;
    }

    Feature sign = (data[i] < Feature(0)) ? Feature(-1) : Feature(1);

    return sign * truncated;
  }
}
