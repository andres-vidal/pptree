#pragma once

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::binarize {
  /**
   * @brief Binarization by largest gap between sorted projected group means.
   *
   * Projects group means onto 1D via the projector, sorts by projected
   * mean, and finds the largest gap between consecutive means. Groups
   * below the gap become binary group 0, groups above become binary
   * group 1. This is the default PPtree binarization method.
   */
  struct LargestGap : public Binarization {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Largest gap"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    /**
     * @brief NodeContext-based interface: binarize and write to ctx.
     */
    void regroup(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: binarize from projected data and partition.
     */
    stats::GroupPartition compute(types::FeatureMatrix const& projected_x, stats::GroupPartition const& y) const;

    static Binarization::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Binarization, "largest_gap")
  };
}
