#pragma once

#include "models/TrainingSpecVisitor.hpp"

#include "models/PPGLDAStrategy.hpp"
#include "models/DRUniformStrategy.hpp"
#include "models/DRNoopStrategy.hpp"


#include <memory>
#include <map>

namespace pptree {
  /**
   * @brief Abstract training configuration for projection pursuit trees.
   *
   * Composes a projection pursuit strategy (PPStrategy) with a
   * dimensionality reduction strategy (DRStrategy).  Concrete
   * subclasses (TrainingSpecGLDA, TrainingSpecUGLDA) provide specific
   * parameter combinations.
   */
  struct TrainingSpec {
    using Ptr = std::unique_ptr<TrainingSpec>;

    /** @brief Projection pursuit optimization strategy. */
    const std::unique_ptr<pp::PPStrategy> pp_strategy;
    /** @brief Dimensionality reduction strategy. */
    const std::unique_ptr<dr::DRStrategy> dr_strategy;

    TrainingSpec(
      std::unique_ptr<pp::PPStrategy> pp_strategy,
      std::unique_ptr<dr::DRStrategy> dr_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)) {
    }

    TrainingSpec(const TrainingSpec& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()) {
    }

    virtual ~TrainingSpec() = default;

    /** @brief Accept a training spec visitor (double dispatch). */
    virtual void accept(TrainingSpecVisitor &visitor) const = 0;

    /** @brief Deep copy of this training specification. */
    virtual Ptr clone() const = 0;
  };
}
