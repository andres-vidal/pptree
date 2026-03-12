#pragma once

namespace ppforest2 {
  struct Tree;
  struct Forest;

  /**
   * @brief Visitor interface for model dispatch.
   *
   * Implements the visitor pattern to distinguish between Tree and
   * Forest models without dynamic_cast.
   */
  struct ModelVisitor {
    virtual void visit(const Tree& tree)     = 0;
    virtual void visit(const Forest& forest) = 0;
  };
}
