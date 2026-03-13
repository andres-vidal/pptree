#pragma once

namespace ppforest2 {
  struct TreeCondition;
  struct TreeResponse;

  /**
   * @brief Visitor interface for tree node dispatch.
   *
   * Implements the visitor pattern to distinguish between internal
   * split nodes (TreeCondition) and leaf nodes (TreeResponse).
   */
  struct TreeNodeVisitor {
    virtual void visit(const TreeCondition &condition) = 0;
    virtual void visit(const TreeResponse &response)   = 0;
  };
}
