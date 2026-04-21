#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"

namespace ppforest2 {
  Model::Ptr Model::train(
      TrainingSpec const& spec,
      types::FeatureMatrix& x,
      types::OutcomeVector& y
  ) {
    // Tree::train / Forest::train return unique_ptr to the abstract base;
    // convert to shared_ptr<Model> for the Model::Ptr interface.
    //
    // Forest::train takes x/y by const ref (bootstraps into per-tree storage
    // internally), so binding our mutable ref to a const-ref parameter is
    // fine — no mutation at the forest level.
    if (spec.is_forest()) {
      return std::shared_ptr<Forest>(Forest::train(spec, x, y).release());
    }

    return std::shared_ptr<Tree>(Tree::train(spec, x, y).release());
  }
}
