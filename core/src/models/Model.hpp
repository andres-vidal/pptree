#pragma once

#include "utils/Types.hpp"

#include <memory>

namespace ppforest2 {
  struct Tree;
  struct Forest;

  /**
   * @brief Abstract base class for predictive models (trees and forests).
   */
  struct Model {
    using Ptr = std::unique_ptr<Model>;

    /**
     * @brief Visitor interface for model dispatch.
     *
     * Implements the visitor pattern to distinguish between Tree and
     * Forest models without dynamic_cast.
     */
    struct Visitor {
      virtual void visit(const Tree& tree)     = 0;
      virtual void visit(const Forest& forest) = 0;
    };

    virtual ~Model() = default;

    /** @brief Accept a model visitor (double dispatch). */
    virtual void accept(Visitor& visitor) const = 0;

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Predicted group label.
     */
    virtual types::Response predict(const types::FeatureVector& data) const = 0;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predicted group labels (n).
     */
    virtual types::ResponseVector predict(const types::FeatureMatrix& data) const = 0;
  };
}
