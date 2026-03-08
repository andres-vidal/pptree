#pragma once

#include "models/TrainingSpec.hpp"

namespace pptree {
  /**
   * @brief UGLDA training specification with uniform variable selection.
   *
   * Combines a GLDA projection pursuit strategy with uniform random
   * dimensionality reduction.  At each split, @c n_vars variables are
   * sampled uniformly and projection pursuit is performed in that
   * subspace.  This is the standard configuration for PP random forests.
   */
  struct TrainingSpecUGLDA : public TrainingSpec {
    /** @brief Number of variables sampled at each split. */
    const int n_vars;
    /** @brief Penalty parameter for the LDA index. */
    const float lambda;

    TrainingSpecUGLDA(const int n_vars, const float lambda) :
      TrainingSpec(pp::glda(lambda), dr::uniform(n_vars)),
      n_vars(n_vars),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpec::Ptr clone() const override {
      return std::make_unique<TrainingSpecUGLDA>(*this);
    }

    static TrainingSpec::Ptr make(const int n_vars, const float lambda) {
      return std::make_unique<TrainingSpecUGLDA>(n_vars, lambda);
    }
  };
}
