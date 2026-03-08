#pragma once

namespace pptree {
  struct TrainingSpecGLDA;
  struct TrainingSpecUGLDA;

  /**
   * @brief Visitor interface for training specification dispatch.
   *
   * Distinguishes between GLDA (all variables) and UGLDA (uniform
   * random variable subset) training configurations.
   */
  struct TrainingSpecVisitor {
    virtual void visit(const TrainingSpecGLDA &spec)  = 0;
    virtual void visit(const TrainingSpecUGLDA &spec) = 0;
  };
}
