#pragma once

namespace pptree {
  struct Tree;
  struct Forest;

  struct ModelVisitor {
    virtual void visit(const Tree& tree)     = 0;
    virtual void visit(const Forest& forest) = 0;
  };
}
