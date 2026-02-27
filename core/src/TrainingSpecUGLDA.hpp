#pragma once

#include "TrainingSpec.hpp"

namespace models {
  struct TrainingSpecUGLDA : public TrainingSpec {
    const int n_vars;
    const float lambda;

    TrainingSpecUGLDA(const int n_vars, const float lambda) :
      TrainingSpec(pp::strategy::glda(lambda), dr::strategy::uniform(n_vars)),
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
