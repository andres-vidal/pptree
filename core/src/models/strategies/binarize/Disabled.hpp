#pragma once

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonReader.hpp"

namespace ppforest2::binarize {
  /**
   * @brief Placeholder binarizer for specs that never reach binarization.
   *
   * Default for regression, where `ByCutpoint` grouping always produces
   * a 2-group partition at each node — so the binary path in tree
   * training never calls `binarize`. `Disabled` gives such specs a
   * well-typed, mode-compatible binarizer without introducing a
   * nullable binarize field.
   *
   * Contract: `regroup` is never called in correct configurations. If
   * it IS called (i.e. an upstream grouping strategy produced a >2-group
   * partition while `Disabled` is configured), the call raises an
   * invariant with a clear diagnostic rather than silently passing
   * through. Catches the misconfiguration at the earliest point
   * instead of letting it propagate to a downstream assertion.
   */
  struct Disabled : public Binarization {
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Disabled"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    /**
     * @brief Always raises an invariant — `Disabled` should never be
     *        reached at runtime. See class-level note.
     */
    void regroup(NodeContext& ctx, stats::RNG& rng) const override;

    static Binarization::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(Binarization, "disabled")
  };

  /** @brief Factory function for the Disabled (placeholder) binarizer. */
  Binarization::Ptr disabled();
}
