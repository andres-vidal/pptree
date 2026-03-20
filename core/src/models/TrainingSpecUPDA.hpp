#pragma once

#include "models/TrainingSpec.hpp"
#include "models/TrainingSpecVisitor.hpp"

namespace ppforest2 {
  /**
   * @brief UPDA training specification with uniform variable selection.
   *
   * Combines a PDA projection pursuit strategy with uniform random
   * dimensionality reduction.  At each split, @c n_vars variables are
   * sampled uniformly and projection pursuit is performed in that
   * subspace.  This is the standard configuration for PP random forests.
   */
  struct TrainingSpecUPDA : public TrainingSpec {
    /** @brief Number of variables sampled at each split. */
    const int n_vars;
    /** @brief Penalty parameter for the LDA index. */
    const float lambda;

    TrainingSpecUPDA(const int n_vars, const float lambda) :
      TrainingSpec(
        pp::pda(lambda),
        dr::uniform(n_vars),
        sr::mean_of_means()),
      n_vars(n_vars),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpec::Ptr clone() const override {
      return std::make_unique<TrainingSpecUPDA>(*this);
    }

    static TrainingSpec::Ptr make(const int n_vars, const float lambda) {
      return std::make_unique<TrainingSpecUPDA>(n_vars, lambda);
    }
  };
}
