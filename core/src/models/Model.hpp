#pragma once

#include "utils/Types.hpp"
#include "models/ModelVisitor.hpp"

namespace pptree {
  struct Model {
    using Ptr = std::unique_ptr<Model>;

    virtual ~Model() = default;

    virtual void accept(ModelVisitor& visitor) const = 0;

    virtual types::Response predict(const types::FeatureVector& data) const       = 0;
    virtual types::ResponseVector predict(const types::FeatureMatrix& data) const = 0;
  };
}
