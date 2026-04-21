#pragma once

#include "models/TrainingSpec.hpp"
#include "utils/Types.hpp"

#include <memory>

namespace ppforest2 {
  struct Tree;
  struct Forest;

  /**
   * @brief Tag type for requesting vote-proportion predictions.
   *
   * For forests, returns an (n × G) matrix of per-tree vote proportions.
   * For single trees, returns an (n × G) one-hot matrix (the tree's
   * deterministic prediction encoded as proportions).
   */
  struct Proportions {};


  /**
   * @brief Abstract base class for predictive models (trees and forests).
   */
  struct Model {
    using Ptr = std::shared_ptr<Model>;

    /**
     * @brief Visitor interface for model dispatch.
     *
     * Implements the visitor pattern to distinguish between Tree and
     * Forest models without dynamic_cast.
     */
    struct Visitor {
      virtual void visit(Tree const& tree)     = 0;
      virtual void visit(Forest const& forest) = 0;
    };

    virtual ~Model() = default;

    /** @brief Whether the model contains degenerate nodes/splits. */
    bool degenerate = false;

    /** @brief Training specification used to build this model. */
    TrainingSpec::Ptr training_spec;

    /** @brief Accept a model visitor (double dispatch). */
    virtual void accept(Visitor& visitor) const = 0;

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Predicted group label.
     */
    virtual types::Outcome predict(types::FeatureVector const& data) const = 0;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predicted group labels (n).
     */
    virtual types::OutcomeVector predict(types::FeatureMatrix const& data) const = 0;

    /**
     * @brief Predict proportions for a matrix of observations.
     *
     * Returns an (n × G) matrix.  For forests, each row contains the
     * fraction of trees that voted for each group.  For single trees,
     * each row is a one-hot encoding of the predicted group.
     *
     * @param data  Feature matrix (n × p).
     * @return      Proportion matrix (n × G), rows sum to 1.0.
     */
    virtual types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const = 0;

    /**
     * @brief Train a model from a training specification.
     *
     * Dispatches to Tree::train or Forest::train based on
     * spec.is_forest().
     *
     * @param spec  Training specification.
     * @param x     Feature matrix (n × p).
     * @param y     Outcome vector (n).
     * @return      Trained model (Tree or Forest).
     */
    static Ptr train(
        TrainingSpec const& spec,
        types::FeatureMatrix& x,
        types::OutcomeVector& y
    );
  };
}
