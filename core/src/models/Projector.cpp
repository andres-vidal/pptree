#include "models/Projector.hpp"
#include "utils/Math.hpp"

using namespace pptree::types;

namespace pptree::pp {
  Projector normalize(const Projector &projector) {
    Projector truncated = (projector.array().abs() < 1e-15).select(0, projector.array());

    const int size      = truncated.size();
    const Feature *data = truncated.data();

    int i = 0;

    while (i < size && math::is_approx(data[i], Feature(0), Feature(0.001))) ++i;

    if (i == size) {
      return truncated;
    }

    Feature sign = (data[i] < Feature(0)) ? Feature(-1) : Feature(1);

    return sign * truncated;
  }
}
