#pragma once

#include "models/PPPDAStrategy.hpp"
#include "models/DRUniformStrategy.hpp"
#include "models/DRNoopStrategy.hpp"
#include "models/SRMeanOfMeansStrategy.hpp"


#include <memory>
#include <map>

namespace ppforest2 {
  struct TrainingSpecPDA;
  struct TrainingSpecUPDA;

  /**
   * @brief Abstract training configuration for projection pursuit trees.
   *
   * Composes a projection pursuit strategy (PPStrategy), a
   * dimensionality reduction strategy (DRStrategy), and a split
   * strategy (SRStrategy).  Concrete subclasses (TrainingSpecPDA,
   * TrainingSpecUPDA) provide specific parameter combinations.
   *
   * For most use cases, use the ready-made subclasses:
   * @code
   *   // Single tree — PDA with all variables:
   *   TrainingSpecPDA spec(lambda: 0.0);
   *
   *   // Random forest — PDA with uniform variable selection:
   *   TrainingSpecUPDA spec(n_vars: 3, lambda: 0.5);
   * @endcode
   *
   * For custom strategy composition:
   * @code
   *   auto spec = std::make_unique<TrainingSpec>(
   *     pp::pda(0.5),          // PDA with lambda = 0.5
   *     dr::uniform(4),         // sample 4 variables per split
   *     sr::mean_of_means());   // midpoint of group means
   * @endcode
   *
   * @see TrainingSpecPDA, TrainingSpecUPDA
   */
  struct TrainingSpec {
    using Ptr = std::unique_ptr<TrainingSpec>;

    /**
     * @brief Visitor interface for training specification dispatch.
     *
     * Distinguishes between PDA (all variables) and UPDA (uniform
     * random variable subset) training configurations.
     */
    struct Visitor {
      virtual void visit(const TrainingSpecPDA &spec)  = 0;
      virtual void visit(const TrainingSpecUPDA &spec) = 0;
    };

    /** @brief Projection pursuit optimization strategy. */
    const std::unique_ptr<pp::PPStrategy> pp_strategy;
    /** @brief Dimensionality reduction strategy. */
    const std::unique_ptr<dr::DRStrategy> dr_strategy;
    /** @brief Group splitting rule strategy. */
    const std::unique_ptr<sr::SRStrategy> sr_strategy;

    TrainingSpec(
      std::unique_ptr<pp::PPStrategy> pp_strategy,
      std::unique_ptr<dr::DRStrategy> dr_strategy,
      std::unique_ptr<sr::SRStrategy> sr_strategy) :
      pp_strategy(std::move(pp_strategy)),
      dr_strategy(std::move(dr_strategy)),
      sr_strategy(std::move(sr_strategy)) {
    }

    TrainingSpec(const TrainingSpec& other) :
      pp_strategy(other.pp_strategy->clone()),
      dr_strategy(other.dr_strategy->clone()),
      sr_strategy(other.sr_strategy->clone()) {
    }

    virtual ~TrainingSpec() = default;

    /** @brief Accept a training spec visitor (double dispatch). */
    virtual void accept(Visitor &visitor) const = 0;

    /** @brief Deep copy of this training specification. */
    virtual Ptr clone() const = 0;
  };
}
