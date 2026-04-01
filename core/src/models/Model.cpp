#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "stats/Stats.hpp"

namespace ppforest2 {
  Model::Ptr Model::train(TrainingSpec const& spec, types::FeatureMatrix const& x, types::ResponseVector const& y) {
    if (spec.is_forest()) {
      return std::make_shared<Forest>(Forest::train(spec, x, y));
    }

    stats::RNG rng(spec.seed);
    return std::make_shared<Tree>(Tree::train(spec, x, y, rng));
  }
}
