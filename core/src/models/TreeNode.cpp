
#include "models/TreeNode.hpp"

namespace pptree {
  bool TreeNode::operator==(const TreeNode& other) const {
    return this->equals(other);
  }

  bool TreeNode::operator!=(const TreeNode& other) const {
    return !this->equals(other);
  }
}
