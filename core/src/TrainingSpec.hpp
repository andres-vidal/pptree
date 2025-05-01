#pragma once

#include "TrainingSpecVisitor.hpp"

#include "PPStrategy.hpp"
#include "DRStrategy.hpp"

#include <memory>
#include <map>

namespace models {
  template<typename T, typename R>
  struct TrainingSpec;
  template<typename T, typename R>
  using TrainingSpecPtr = std::unique_ptr<TrainingSpec<T, R> >;

  template<typename T, typename R>
  struct TrainingSpec {
    const std::unique_ptr<pp::strategy::PPStrategy<T, R> > pp_strategy;
    const std::unique_ptr<dr::strategy::DRStrategy<T, R> > dr_strategy;

    TrainingSpec(
      std::unique_ptr<pp::strategy::PPStrategy<T, R> > pp_strategy,
      std::unique_ptr<dr::strategy::DRStrategy<T, R> > dr_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)) {
    }

    TrainingSpec(const TrainingSpec<T, R>& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()) {
    }

    virtual ~TrainingSpec()                                       = default;
    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const = 0;

    virtual TrainingSpecPtr<T, R> clone() const = 0;

    static TrainingSpecPtr<T, R> glda(const float lambda);
    static TrainingSpecPtr<T, R> lda();
    static TrainingSpecPtr<T, R> uniform_glda(const int n_vars, const float lambda);
  };
  template<typename T, typename R>
  struct TrainingSpecGLDA : public TrainingSpec<T, R> {
    const float lambda;

    explicit TrainingSpecGLDA(const float lambda) :
      TrainingSpec<T, R>(
        pp::strategy::glda<T, R>(lambda),
        dr::strategy::all<T, R>()),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpecPtr<T, R> clone() const override {
      return std::make_unique<TrainingSpecGLDA<T, R> >(*this);
    }
  };

  template<typename T, typename R>
  struct TrainingSpecUGLDA : public TrainingSpec<T, R> {
    const int n_vars;
    const float lambda;

    TrainingSpecUGLDA(const int n_vars, const float lambda) :
      TrainingSpec<T, R>(
        pp::strategy::glda<T, R>(lambda),
        dr::strategy::uniform<T, R>(n_vars)),
      n_vars(n_vars),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    TrainingSpecPtr<T, R> clone() const override {
      return std::make_unique<TrainingSpecUGLDA<T, R> >(*this);
    }
  };


  template<typename T, typename R>
  TrainingSpecPtr<T, R> TrainingSpec<T, R>::glda(const float lambda) {
    return std::make_unique<TrainingSpecGLDA<T, R> >(lambda);
  }

  template<typename T, typename R>
  TrainingSpecPtr<T, R> TrainingSpec<T, R>::lda() {
    return std::make_unique<TrainingSpecGLDA<T, R> >(0.0);
  }

  template<typename T, typename R>
  TrainingSpecPtr<T, R> TrainingSpec<T, R>::uniform_glda(const int n_vars, const float lambda) {
    return std::make_unique<TrainingSpecUGLDA<T, R> >(n_vars, lambda);
  }
}
