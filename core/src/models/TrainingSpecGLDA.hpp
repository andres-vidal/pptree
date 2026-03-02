#pragma once

#include "models/TrainingSpec.hpp"

namespace pptree {
  struct TrainingSpecGLDA : public TrainingSpec {
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
