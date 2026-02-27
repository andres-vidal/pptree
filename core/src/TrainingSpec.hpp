#pragma once

#include "TrainingSpecVisitor.hpp"

#include "PPGLDAStrategy.hpp"
#include "DRUniformStrategy.hpp"
#include "DRNoopStrategy.hpp"


#include <memory>
#include <map>

namespace models {
  struct TrainingSpec {
    using Ptr = std::unique_ptr<TrainingSpec>;


    const std::unique_ptr<pp::strategy::PPStrategy> pp_strategy;
    const std::unique_ptr<dr::strategy::DRStrategy> dr_strategy;

    TrainingSpec(
      std::unique_ptr<pp::strategy::PPStrategy> pp_strategy,
      std::unique_ptr<dr::strategy::DRStrategy> dr_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)) {
    }

    TrainingSpec(const TrainingSpec& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()) {
    }

    virtual ~TrainingSpec()                                 = default;
    virtual void accept(TrainingSpecVisitor &visitor) const = 0;

    virtual Ptr clone() const = 0;
  };
}
