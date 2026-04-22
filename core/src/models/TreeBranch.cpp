#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "utils/Math.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::pp;

namespace ppforest2 {
  TreeBranch::TreeBranch(
      Projector projector,
      Feature cutpoint,
      TreeNode::Ptr lower,
      TreeNode::Ptr upper,
      std::set<GroupId> groups,
      Feature pp_index_value
  )
      : projector(std::move(projector))
      , cutpoint(cutpoint)
      , lower(std::move(lower))
      , upper(std::move(upper))
      , groups(std::move(groups))
      , pp_index_value(pp_index_value) {
    degenerate = this->lower->degenerate || this->upper->degenerate;
  }

  void TreeBranch::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Outcome TreeBranch::response() const {
    throw std::invalid_argument("Cannot get response from a condition node");
  }

  Outcome TreeBranch::predict(FeatureVector const& data) const {
    Feature const projected = data.dot(projector);

    if (projected < cutpoint) {
      return lower->predict(data);
    }

    return upper->predict(data);
  }

  bool TreeBranch::equals(TreeNode const& other) const {
    auto const* cond = dynamic_cast<TreeBranch const*>(&other);

    // Intentionally structural equality (metadata ignored).
    return (cond != nullptr) && math::collinear(projector, cond->projector) &&
           math::is_approx(cutpoint, cond->cutpoint) && *lower == *(cond->lower) && *upper == *(cond->upper);
  }

  TreeNode::Ptr TreeBranch::clone() const {
    return make(projector, cutpoint, lower->clone(), upper->clone(), groups, pp_index_value);
  }

  TreeBranch::Ptr TreeBranch::make(
      Projector projector,
      Feature cutpoint,
      TreeNode::Ptr lower,
      TreeNode::Ptr upper,
      std::set<GroupId> groups,
      Feature pp_index_value
  ) {
    return std::make_unique<TreeBranch>(
        std::move(projector), cutpoint, std::move(lower), std::move(upper), std::move(groups), pp_index_value
    );
  }
}
