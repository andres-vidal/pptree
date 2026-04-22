#include "models/strategies/stop/MinVariance.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace ppforest2::types;

namespace ppforest2::stop {
  MinVariance::MinVariance(Feature threshold)
      : threshold(threshold) {}

  nlohmann::json MinVariance::to_json() const {
    return {{"name", "min_variance"}, {"threshold", threshold}};
  }

  std::string MinVariance::display_name() const {
    std::ostringstream oss;
    // `defaultfloat` picks fixed or scientific notation automatically so
    // small thresholds like 1e-6 render as "1e-06" instead of "0.0000".
    // Cap precision at 6 significant digits — the threshold is a `Feature`
    // (float) so more digits would just print noise.
    oss << "Min variance (" << std::defaultfloat << std::setprecision(6) << threshold << ")";
    return oss.str();
  }

  bool MinVariance::should_stop(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    invariant(ctx.y_vec != nullptr, "MinVariance requires y_vec in NodeContext");

    // Subset of y for observations in this node, evaluated in double precision.
    // Accumulating in `float` is non-associative and unstable under -O2
    // compiler reorderings (CLAUDE.md allows internal double casts). Eigen's
    // reductions have a fixed traversal order determined by the build-time
    // SIMD baseline (the project compiles without -march=native, guaranteeing
    // a consistent SIMD width across platforms), so the result is
    // cross-platform reproducible.
    Eigen::VectorXd const y = ctx.y.data(*ctx.y_vec).template cast<double>();

    int const count = static_cast<int>(y.size());
    if (count <= 1) {
      return true;
    }

    double const mean     = y.mean();
    double const variance = (y.array() - mean).square().sum() / static_cast<double>(count - 1);

    return variance < static_cast<double>(threshold);
  }

  StopRule::Ptr min_variance(Feature threshold) {
    return std::make_shared<MinVariance>(threshold);
  }

  StopRule::Ptr MinVariance::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "min_variance"};
    r.only_keys({"name", "threshold"});
    return min_variance(static_cast<Feature>(r.require_number("threshold", 0.0)));
  }
}
