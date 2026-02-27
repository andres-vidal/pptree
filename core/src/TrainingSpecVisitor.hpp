#pragma once

namespace models {
  struct TrainingSpecGLDA;
  struct TrainingSpecUGLDA;

  struct TrainingSpecVisitor {
    virtual void visit(const TrainingSpecGLDA &spec)  = 0;
    virtual void visit(const TrainingSpecUGLDA &spec) = 0;
  };
}
