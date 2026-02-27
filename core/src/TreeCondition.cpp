#include "TreeCondition.hpp"
#include "TreeResponse.hpp"

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace models::types;
using namespace models::pp;
using namespace models::stats;

namespace models {
  TreeCondition::TreeCondition(
    Projector          projector,
    Threshold          threshold,
    TreeNode::Ptr      lower,
    TreeNode::Ptr      upper,
    TrainingSpec::Ptr  training_spec,
    std::set<Response> classes)
    : projector(std::move(projector)),
    threshold(std::move(threshold)),
    lower(std::move(lower)),
    upper(std::move(upper)),
    training_spec(std::move(training_spec)),
    classes(std::move(classes)) {
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

  json TreeCondition::to_json() const {
    // Keep JSON stable / minimal; metadata stays in-memory unless you explicitly add it.
    return json{
      { "projector", projector },
      { "threshold", threshold },
      { "lower", lower->to_json() },
      { "upper", upper->to_json() },
    };
  }

  TreeNode::Ptr TreeCondition::clone() const {
    TrainingSpec::Ptr spec_clone = training_spec ? training_spec->clone() : nullptr;

    return make(
      projector,
      threshold,
      lower->clone(),
      upper->clone(),
      std::move(spec_clone),
      classes);
  }

  TreeCondition::Ptr TreeCondition::make(
    Projector          projector,
    Threshold          threshold,
    TreeNode::Ptr      lower,
    TreeNode::Ptr      upper,
    TrainingSpec::Ptr  training_spec,
    std::set<Response> classes) {
    return std::make_unique<TreeCondition>(
      std::move(projector),
      std::move(threshold),
      std::move(lower),
      std::move(upper),
      std::move(training_spec),
      std::move(classes)
      );
  }

  std::ostream& operator<<(std::ostream& ostream, const TreeCondition& condition) {
    return ostream << condition.to_json().dump(2, ' ', false);
  }
}
