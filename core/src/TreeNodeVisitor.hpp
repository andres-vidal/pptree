#pragma once

namespace models {
  struct TreeCondition;

  struct TreeResponse;

  struct TreeNodeVisitor {
    virtual void visit(const TreeCondition &condition) = 0;
    virtual void visit(const TreeResponse &response)   = 0;
  };
}
