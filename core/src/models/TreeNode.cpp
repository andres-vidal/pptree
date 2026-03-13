
#include "models/TreeNode.hpp"

namespace ppforest2 {
  bool TreeNode::operator==(const TreeNode& other) const {
    return this->equals(other);
  }

  bool TreeNode::operator!=(const TreeNode& other) const {
    return !this->equals(other);
  }
}
