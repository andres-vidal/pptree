#pragma once

#include "TrainingSpec.hpp"

namespace models {
  template<typename T, typename R>
  struct TrainingSpecUGLDA : public TrainingSpec<T, R> {
    const int n_vars;
    const float lambda;

    TrainingSpecUGLDA(const int n_vars, const float lambda) :
      TrainingSpec<T, R>(pp::strategy::glda<T, R>(lambda), dr::strategy::uniform<T, R>(n_vars)),
      n_vars(n_vars),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpecPtr<T, R> clone() const override {
      return std::make_unique<TrainingSpecUGLDA<T, R> >(*this);
    }

    static TrainingSpecPtr<T, R> make(const int n_vars, const float lambda) {
      return std::make_unique<TrainingSpecUGLDA<T, R> >(n_vars, lambda);
    }
  };
}
