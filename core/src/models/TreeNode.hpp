#pragma once

#include <memory>

#include "utils/Types.hpp"
#include "models/TreeNodeVisitor.hpp"

namespace pptree {
  struct TreeNode {
    using Ptr = std::unique_ptr<TreeNode>;

    virtual ~TreeNode()                                                     = default;
    virtual void accept(TreeNodeVisitor &visitor) const                     = 0;
    virtual types::Response predict(const types::FeatureVector &data) const = 0;
    virtual types::Response response() const                                = 0;

    virtual bool equals(const TreeNode &other) const = 0;
    virtual Ptr clone() const                        = 0;

    bool operator==(const TreeNode &other) const;
    bool operator!=(const TreeNode &other) const;
  };
}
