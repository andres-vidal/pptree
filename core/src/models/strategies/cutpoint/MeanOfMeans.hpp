#pragma once

#include "models/Projector.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"
#include "utils/Types.hpp"

namespace ppforest2::cutpoint {
  /**
   * @brief Split cutpoint as the mean of two group means.
   *
   * Computes the midpoint between the projected means of the two
   * groups: (mean(group_1 * A) + mean(group_2 * A)) / 2.
   * This is the default rule used by PPforest.
   */
  struct MeanOfMeans : public Cutpoint {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Mean of means"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    /**
     * @brief NodeContext-based interface: compute cutpoint and write to ctx.cutpoint.
     */
    void cutpoint(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: compute cutpoint from two group matrices and projector.
     */
    types::Feature compute(
        types::FeatureMatrix const& group_1,
        types::FeatureMatrix const& group_2,
        ppforest2::pp::Projector const& projector
    ) const;

    static Cutpoint::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Cutpoint, "mean_of_means")
  };
}
