#pragma once

namespace ppforest2 {
  struct TrainingSpecPDA;
  struct TrainingSpecUPDA;

  /**
   * @brief Visitor interface for training specification dispatch.
   *
   * Distinguishes between PDA (all variables) and UPDA (uniform
   * random variable subset) training configurations.
   */
  struct TrainingSpecVisitor {
    virtual void visit(const TrainingSpecPDA &spec)  = 0;
    virtual void visit(const TrainingSpecUPDA &spec) = 0;
  };
}
