#pragma once

#include "PPStrategy.hpp"
#include "DRStrategy.hpp"

#include <memory>
#include <map>

namespace models {
  template<typename T, typename R>
  struct GLDATrainingSpec;
  template<typename T, typename R>
  struct UniformGLDATrainingSpec;

  template <typename T, typename R>
  struct TrainingSpecVisitor {
    virtual void visit(const GLDATrainingSpec<T, R> &spec) = 0;
    virtual void visit(const UniformGLDATrainingSpec<T, R> &spec) = 0;
  };

  template<typename T, typename R>
  struct TrainingSpec {
    const std::unique_ptr<pp::strategy::PPStrategy<T, R> > pp_strategy;
    const std::unique_ptr<dr::strategy::DRStrategy<T> > dr_strategy;

    TrainingSpec(
      std::unique_ptr<pp::strategy::PPStrategy<T, R> > pp_strategy,
      std::unique_ptr<dr::strategy::DRStrategy<T> >    dr_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)) {
    }

    TrainingSpec(const TrainingSpec& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()) {
    }

    virtual ~TrainingSpec() = default;
    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const = 0;

    virtual std::unique_ptr<TrainingSpec<T, R> > clone() const = 0;

    static std::unique_ptr<TrainingSpec<T, R> > glda(const double lambda);
    static std::unique_ptr<TrainingSpec<T, R> > lda();
    static std::unique_ptr<TrainingSpec<T, R> > uniform_glda(const int n_vars, const double lambda);
  };

  template<typename T, typename R>
  struct GLDATrainingSpec : public TrainingSpec<T, R> {
    const double lambda;

    explicit GLDATrainingSpec(const double lambda) :
      TrainingSpec<T, R>(
        pp::strategy::glda<T, R>(lambda),
        dr::strategy::all<T>()),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    std::unique_ptr<TrainingSpec<T, R> > clone() const override {
      return std::make_unique<GLDATrainingSpec<T, R> >(*this);
    }
  };

  template<typename T, typename R>
  struct UniformGLDATrainingSpec : public TrainingSpec<T, R> {
    const int n_vars;
    const double lambda;

    UniformGLDATrainingSpec(const int n_vars, const double lambda) :
      TrainingSpec<T, R>(
        pp::strategy::glda<T, R>(lambda),
        dr::strategy::uniform<T>(n_vars)),
      n_vars(n_vars),
      lambda(lambda) {
    }

    virtual void accept(TrainingSpecVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    std::unique_ptr<TrainingSpec<T, R> > clone() const override {
      return std::make_unique<UniformGLDATrainingSpec<T, R> >(*this);
    }
  };


  template<typename T, typename R>
  std::unique_ptr<TrainingSpec<T, R> > TrainingSpec<T, R>::glda(const double lambda) {
    return std::make_unique<GLDATrainingSpec<T, R> >(lambda);
  }

  template<typename T, typename R>
  std::unique_ptr<TrainingSpec<T, R> > TrainingSpec<T, R>::lda() {
    return std::make_unique<GLDATrainingSpec<T, R> >(0.0);
  }

  template<typename T, typename R>
  std::unique_ptr<TrainingSpec<T, R> >TrainingSpec<T, R>::uniform_glda(const int n_vars, const double lambda) {
    return std::make_unique<UniformGLDATrainingSpec<T, R> >(n_vars, lambda);
  }
}
