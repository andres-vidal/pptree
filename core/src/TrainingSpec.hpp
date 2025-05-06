#pragma once

#include "TrainingSpecVisitor.hpp"

#include "PPGLDAStrategy.hpp"
#include "DRUniformStrategy.hpp"
#include "DRNoopStrategy.hpp"


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
  };
}
