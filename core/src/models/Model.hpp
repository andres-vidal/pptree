#pragma once

#include "utils/Types.hpp"

#include <memory>

namespace pptree {
  struct ModelVisitor;
  /**
   * @brief Abstract base class for predictive models (trees and forests).
   */
  struct Model {
    using Ptr = std::unique_ptr<Model>;

    virtual ~Model() = default;

    /** @brief Accept a model visitor (double dispatch). */
    virtual void accept(ModelVisitor& visitor) const = 0;

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Predicted class label.
     */
    virtual types::Response predict(const types::FeatureVector& data) const = 0;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predicted class labels (n).
     */
    virtual types::ResponseVector predict(const types::FeatureMatrix& data) const = 0;
  };
}
