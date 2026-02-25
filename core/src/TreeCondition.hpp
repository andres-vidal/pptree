#pragma once

#include "TreeNode.hpp"

namespace models {
  template<typename T, typename R>
  using TreeConditionPtr = std::unique_ptr<TreeCondition<T, R> >;
  template<typename T, typename R>
  struct TreeCondition : public TreeNode<T, R> {
    pp::Projector<T> projector;
    T threshold;
    TreeNodePtr<T, R> lower;
    TreeNodePtr<T, R> upper;
    TrainingSpecPtr<T, R> training_spec;

    const std::set<R> classes;

    TreeCondition(
      const pp::Projector<T>& projector,
      const Threshold<T>&     threshold,
      TreeNodePtr<T, R>       lower,
      TreeNodePtr<T, R>       upper,
      TrainingSpecPtr<T, R>   training_spec,
      const std::set<R> &     classes) :
      projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)),
      training_spec(std::move(training_spec)),
      classes(classes) {
    }

    TreeCondition(
      const pp::Projector<T>& projector,
      const Threshold<T>&     threshold,
      TreeNodePtr<T, R>       lower,
      TreeNodePtr<T, R>       upper) :
      projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)) {
    }

    void accept(TreeNodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      throw std::invalid_argument("Cannot get response from a condition node");
    }

    R predict(const stats::DataColumn<T> &data) const override {
      T projected_data = data.dot(projector);

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }

    bool equals(const TreeNode<T, R> &other) const override {
      const auto *cond = dynamic_cast<const TreeCondition<T, R> *>(&other);

      return cond
             && math::collinear(projector, cond->projector)
             && math::is_approx(threshold, cond->threshold)
             && *lower == *(cond->lower)
             && *upper == *(cond->upper);
    }

    json to_json() const override {
      return json{
        { "projector", projector },
        { "threshold", threshold },
        { "lower", lower->to_json() },
        { "upper", upper->to_json() }
      };
    }

    TreeNodePtr<T, R> clone() const override {
      return make(projector, threshold, lower->clone(), upper->clone());
    }

    static TreeConditionPtr<T, R> make(
      const pp::Projector<T>& projector,
      const Threshold<T>&     threshold,
      TreeNodePtr<T, R>       lower,
      TreeNodePtr<T, R>       upper,
      TrainingSpecPtr<T, R>   training_spec,
      const std::set<R> &     classes) {
      return std::make_unique<TreeCondition<T, R> >(
        projector,
        threshold,
        std::move(lower),
        std::move(upper),
        std::move(training_spec),
        classes);
    }

    static TreeConditionPtr<T, R> make(
      const pp::Projector<T>& projector,
      const Threshold<T>&     threshold,
      TreeNodePtr<T, R>       lower,
      TreeNodePtr<T, R>       upper) {
      return std::make_unique<TreeCondition<T, R> >(
        projector,
        threshold,
        std::move(lower),
        std::move(upper));
    }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const TreeCondition<T, R>& condition) {
    return ostream << condition.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  TreeNodePtr<T, R> node_from_json(const json& j) {
    if (j.contains("value")) {
      return TreeResponse<T, R>::make(j["value"].get<R>());
    }

    auto proj_vec              = j["projector"].get<std::vector<T> >();
    pp::Projector<T> projector = Eigen::Map<pp::Projector<T> >(proj_vec.data(), proj_vec.size());
    T threshold                = j["threshold"].get<T>();
    auto lower                 = node_from_json<T, R>(j["lower"]);
    auto upper                 = node_from_json<T, R>(j["upper"]);

    return TreeCondition<T, R>::make(projector, threshold, std::move(lower), std::move(upper));
  }
}
