#pragma once

/**
 * @file TestSpec.hpp
 * @brief Shared test helpers for constructing throwaway `TrainingSpec`s.
 *
 * Many structural tests construct a `Tree` or `Forest` directly from a
 * hand-built `TreeNode` graph — they exercise traversal, equality, and
 * prediction without caring about the training configuration. These helpers
 * provide a minimal, mode-correct `TrainingSpec::Ptr` so the subclass/mode
 * invariants on `ClassificationTree`, `RegressionTree`,
 * `ClassificationForest`, and `RegressionForest` are always satisfied.
 *
 * Not intended for production use — production callers get their spec from
 * training or deserialization.
 */

#include "models/TrainingSpec.hpp"
#include "models/strategies/grouping/ByCutpoint.hpp"
#include "models/strategies/leaf/MeanResponse.hpp"
#include "models/strategies/stop/CompositeStop.hpp"
#include "models/strategies/stop/MinSize.hpp"
#include "models/strategies/stop/MinVariance.hpp"

namespace ppforest2::test {
  /**
   * @brief Default-valued classification spec, as a shared pointer.
   *
   * All strategy defaults (PDA lambda=0, `vars::all()`, pure-node stop, etc.)
   * are irrelevant for structural tests but need to be well-formed.
   */
  inline TrainingSpec::Ptr classification_spec() {
    return TrainingSpec::builder(types::Mode::Classification).make();
  }

  /**
   * @brief Default-valued regression spec, as a shared pointer.
   *
   * Uses `by_cutpoint` grouping, `mean_response` leaves, and a regression-
   * compatible stop rule. The builder's default stop (`pure_node`) is
   * classification-only, so leaving it at the default here would throw
   * `std::invalid_argument` out of `TrainingSpec`'s mode-compat check.
   *
   * Stop rule `any({min_size(2), min_variance(0.0)})` is deliberately
   * loose: `min_size(2)` is the minimum useful threshold, and `0.0F` is
   * a sentinel "fire only on identically-constant nodes" variance bound.
   * The intent is a semantically-minimal, spec-valid stop rule for
   * structural tests that never actually train. Tests that need a
   * realistic regression stop should construct their own spec.
   */
  inline TrainingSpec::Ptr regression_spec() {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::any({stop::min_size(2), stop::min_variance(0.0F)}))
        .make();
  }
}
