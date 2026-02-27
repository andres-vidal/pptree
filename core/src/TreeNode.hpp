#pragma once

#include "Types.hpp"
#include "TreeNodeVisitor.hpp"

#include <nlohmann/json.hpp>
#include <iosfwd>

namespace models {
  using json      = nlohmann::json;
  using Threshold = types::Feature;

  struct TreeNode {
    using Ptr = std::unique_ptr<TreeNode>;

    virtual ~TreeNode()                                                     = default;
    virtual void accept(TreeNodeVisitor &visitor) const                     = 0;
    virtual types::Response predict(const types::FeatureVector &data) const = 0;
    virtual types::Response response() const                                = 0;
    virtual json to_json() const                                            = 0;

    virtual bool equals(const TreeNode &other) const = 0;
    virtual Ptr clone() const                        = 0;

    bool operator==(const TreeNode &other) const;
    bool operator!=(const TreeNode &other) const;
  };

  std::ostream& operator<<(std::ostream& ostream, const TreeNode& node);
  TreeNode::Ptr node_from_json(const json& j);
}
