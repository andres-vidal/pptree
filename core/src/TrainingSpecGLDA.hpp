#pragma once

#include "TrainingSpec.hpp"

namespace models {
  template<typename T, typename R>
  struct TrainingSpecGLDA : public TrainingSpec<T, R> {
    const float lambda;

    explicit TrainingSpecGLDA(const float lambda) :
      TrainingSpec<T, R>(
        pp::strategy::glda<T, R>(lambda),
        dr::strategy::noop<T, R>()),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpecPtr<T, R> clone() const override {
      return std::make_unique<TrainingSpecGLDA<T, R> >(*this);
    }

    static TrainingSpecPtr<T, R> make(const float lambda) {
      return std::make_unique<TrainingSpecGLDA<T, R> >(lambda);
    }
  };
}
