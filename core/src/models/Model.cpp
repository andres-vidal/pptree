#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"

namespace ppforest2 {
  Model::Ptr Model::train(TrainingSpec const& spec, types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    if (spec.is_forest()) {
      return std::make_shared<Forest>(Forest::train(spec, x, y));
    }

    return std::make_shared<Tree>(Tree::train(spec, x, y));
  }
}
