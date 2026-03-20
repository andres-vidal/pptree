#pragma once

#include "models/TrainingSpec.hpp"

namespace ppforest2 {
  /**
   * @brief PDA training specification using all variables.
   *
   * Combines a PDA projection pursuit strategy with a no-op
   * dimensionality reduction (all variables are used at every split).
   * This is the standard configuration for a single PP tree.
   */
  struct TrainingSpecPDA : public TrainingSpec {
    /** @brief Penalty parameter for the LDA index. */
    const float lambda;

    explicit TrainingSpecPDA(const float lambda) :
      TrainingSpec(
        pp::pda(lambda),
        dr::noop(),
        sr::mean_of_means()),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpec::Visitor &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpec::Ptr clone() const override {
      return std::make_unique<TrainingSpecPDA>(*this);
    }

    static TrainingSpec::Ptr make(const float lambda) {
      return std::make_unique<TrainingSpecPDA>(lambda);
    }
  };
}
