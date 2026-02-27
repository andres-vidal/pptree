#pragma once

#include "TrainingSpec.hpp"

namespace models {
  struct TrainingSpecGLDA : public TrainingSpec {
    const float lambda;

    explicit TrainingSpecGLDA(const float lambda) :
      TrainingSpec(
        pp::strategy::glda(lambda),
        dr::strategy::noop()),
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
