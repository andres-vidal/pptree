namespace models {
  template<typename T, typename R>
  struct TreeCondition;

  template<typename T, typename R>
  struct TreeResponse;

  template<typename T, typename R>
  struct TreeNodeVisitor {
    virtual void visit(const TreeCondition<T, R> &condition) = 0;
    virtual void visit(const TreeResponse<T, R> &response)   = 0;
  };
}
