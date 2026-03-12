#pragma once

#include "models/PPGLDAStrategy.hpp"
#include "models/DRUniformStrategy.hpp"
#include "models/DRNoopStrategy.hpp"
#include "models/SRMeanOfMeansStrategy.hpp"


#include <memory>
#include <map>

namespace ppforest2 {
  struct TrainingSpecVisitor;
  /**
   * @brief Abstract training configuration for projection pursuit trees.
   *
   * Composes a projection pursuit strategy (PPStrategy), a
   * dimensionality reduction strategy (DRStrategy), and a split
   * strategy (SRStrategy).  Concrete subclasses (TrainingSpecGLDA,
   * TrainingSpecUGLDA) provide specific parameter combinations.
   */
  struct TrainingSpec {
    using Ptr = std::unique_ptr<TrainingSpec>;

    /** @brief Projection pursuit optimization strategy. */
    const std::unique_ptr<pp::PPStrategy> pp_strategy;
    /** @brief Dimensionality reduction strategy. */
    const std::unique_ptr<dr::DRStrategy> dr_strategy;
    /** @brief Group splitting rule strategy. */
    const std::unique_ptr<sr::SRStrategy> split_strategy;

    TrainingSpec(
      std::unique_ptr<pp::PPStrategy> pp_strategy,
      std::unique_ptr<dr::DRStrategy> dr_strategy,
      std::unique_ptr<sr::SRStrategy> split_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)),
      split_strategy(std::move(split_strategy)) {
    }

    TrainingSpec(const TrainingSpec& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()),
      split_strategy(other.split_strategy->clone()) {
    }

    virtual ~TrainingSpec() = default;

    /** @brief Accept a training spec visitor (double dispatch). */
    virtual void accept(TrainingSpecVisitor &visitor) const = 0;

    /** @brief Deep copy of this training specification. */
    virtual Ptr clone() const = 0;
  };
}
