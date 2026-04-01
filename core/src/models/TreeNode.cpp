
#include "models/TreeNode.hpp"

namespace ppforest2 {
  bool TreeNode::operator==(TreeNode const& other) const {
    return this->equals(other);
  }

  bool TreeNode::operator!=(TreeNode const& other) const {
    return !this->equals(other);
  }
}
