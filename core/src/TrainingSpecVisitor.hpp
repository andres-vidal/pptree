#pragma once

namespace models {
  template<typename T, typename R> struct TrainingSpecGLDA;
  template<typename T, typename R> struct TrainingSpecUGLDA;

  template <typename T, typename R>
  struct TrainingSpecVisitor {
    virtual void visit(const TrainingSpecGLDA<T, R> &spec)  = 0;
    virtual void visit(const TrainingSpecUGLDA<T, R> &spec) = 0;
  };
}
