#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::pp;

namespace ppforest2 {
  TreeCondition::TreeCondition(Projector projector,
                               Threshold threshold,
                               TreeNode::Ptr lower,
                               TreeNode::Ptr upper,
                               std::set<Response> groups,
                               Feature pp_index_value)
      : projector(std::move(projector))
      , threshold(std::move(threshold))
      , lower(std::move(lower))
      , upper(std::move(upper))
      , groups(std::move(groups))
      , pp_index_value(pp_index_value) {
    degenerate = this->lower->degenerate || this->upper->degenerate;
  }

  void TreeCondition::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Response TreeCondition::response() const {
    throw std::invalid_argument("Cannot get response from a condition node");
  }

  Response TreeCondition::predict(FeatureVector const& data) const {
    Feature const projected = data.dot(projector);

    if (projected < threshold) {
      return lower->predict(data);
    }

    return upper->predict(data);
  }

  bool TreeCondition::equals(TreeNode const& other) const {
    auto const* cond = dynamic_cast<TreeCondition const*>(&other);

    // Intentionally structural equality (metadata ignored).
    return cond && math::collinear(projector, cond->projector) && math::is_approx(threshold, cond->threshold) &&
           *lower == *(cond->lower) && *upper == *(cond->upper);
  }

  TreeNode::Ptr TreeCondition::clone() const {
    return make(projector, threshold, lower->clone(), upper->clone(), groups, pp_index_value);
  }

  TreeCondition::Ptr TreeCondition::make(Projector projector,
                                         Threshold threshold,
                                         TreeNode::Ptr lower,
                                         TreeNode::Ptr upper,
                                         std::set<Response> groups,
                                         Feature pp_index_value) {
    return std::make_unique<TreeCondition>(std::move(projector),
                                           std::move(threshold),
                                           std::move(lower),
                                           std::move(upper),
                                           std::move(groups),
                                           pp_index_value);
  }
}
