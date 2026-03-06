#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace pptree::types;
using namespace pptree::pp;
using namespace pptree::stats;

namespace pptree {
  TreeCondition::TreeCondition(
    Projector          projector,
    Threshold          threshold,
    TreeNode::Ptr      lower,
    TreeNode::Ptr      upper,
    TrainingSpec::Ptr  training_spec,
    std::set<Response> classes,
    Feature            pp_index_value) :
    projector(std::move(projector)),
    threshold(std::move(threshold)),
    lower(std::move(lower)),
    upper(std::move(upper)),
    training_spec(std::move(training_spec)),
    classes(std::move(classes)),
    pp_index_value(pp_index_value) {
  }

  void TreeCondition::accept(TreeNodeVisitor& visitor) const {
    visitor.visit(*this);
  }

  Response TreeCondition::response() const {
    throw std::invalid_argument("Cannot get response from a condition node");
  }

  Response TreeCondition::predict(const FeatureVector& data) const {
    const Feature projected = data.dot(projector);

    if (projected < threshold) {
      return lower->predict(data);
    }

    return upper->predict(data);
  }

  bool TreeCondition::equals(const TreeNode& other) const {
    const auto *cond = dynamic_cast<const TreeCondition *>(&other);

    // Intentionally structural equality (metadata ignored).
    return cond
           && math::collinear(projector, cond->projector)
           && math::is_approx(threshold, cond->threshold)
           && *lower == *(cond->lower)
           && *upper == *(cond->upper);
  }

  TreeNode::Ptr TreeCondition::clone() const {
    TrainingSpec::Ptr spec_clone = training_spec ? training_spec->clone() : nullptr;

    return make(
      projector,
      threshold,
      lower->clone(),
      upper->clone(),
      std::move(spec_clone),
      classes,
      pp_index_value);
  }

  TreeCondition::Ptr TreeCondition::make(
    Projector          projector,
    Threshold          threshold,
    TreeNode::Ptr      lower,
    TreeNode::Ptr      upper,
    TrainingSpec::Ptr  training_spec,
    std::set<Response> classes,
    Feature            pp_index_value) {
    return std::make_unique<TreeCondition>(
      std::move(projector),
      std::move(threshold),
      std::move(lower),
      std::move(upper),
      std::move(training_spec),
      std::move(classes),
      pp_index_value
      );
  }
}
