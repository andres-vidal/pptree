#pragma once

#include "models/TrainingSpec.hpp"

namespace pptree {
  /**
   * @brief GLDA training specification using all variables.
   *
   * Combines a GLDA projection pursuit strategy with a no-op
   * dimensionality reduction (all variables are used at every split).
   * This is the standard configuration for a single PP tree.
   */
  struct TrainingSpecGLDA : public TrainingSpec {
    /** @brief Penalty parameter for the LDA index. */
    const float lambda;

    explicit TrainingSpecGLDA(const float lambda) :
      TrainingSpec(
        pp::glda(lambda),
        dr::noop()),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpec::Ptr clone() const override {
      return std::make_unique<TrainingSpecGLDA>(*this);
    }

    static TrainingSpec::Ptr make(const float lambda) {
      return std::make_unique<TrainingSpecGLDA>(lambda);
    }
  };
}
