#pragma once

#include "models/Model.hpp"
#include "models/BootstrapTree.hpp"

#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace pptree {
  /**
   * @brief Tag type for requesting vote-proportion predictions.
   *
   * Pass an instance of this type to Forest::predict to get an (n × G)
   * matrix of vote proportions instead of a single majority-vote response.
   */
  struct Proportions {};

  struct Forest : public Model {
    /**
     * @brief Train a random forest.
     *
     * @param training_spec  Training specification.
     * @param x              Feature matrix (n × p).
     * @param y              Response vector (n).
     * @param size           Number of trees.
     * @param seed            RNG seed.
     * @param n_threads      Number of threads (default: hardware concurrency).
     * @return               Trained forest.
     */
    static Forest train(
      const TrainingSpec&          training_spec,
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y,
      int                          size,
      int                          seed,
      int                          n_threads = std::thread::hardware_concurrency());

    std::vector<BootstrapTree::Ptr> trees;
    TrainingSpec::Ptr training_spec;
    const int seed = 0;

    Forest();
    Forest(TrainingSpec::Ptr&& training_spec, int seed);

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Prediction.
     */
    types::Response predict(const types::FeatureVector& data) const override;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predictions (n).
     */
    types::ResponseVector predict(const types::FeatureMatrix& data) const override;

    /**
     * @brief Predict vote proportions for a matrix of observations.
     *
     * For each observation, counts votes from every tree and returns
     * the proportion of trees that voted for each class.  The number
     * of classes G is derived from the root node of the first tree.
     *
     * @param data  Feature matrix (n × p).
     * @return      Vote proportions matrix (n × G), rows sum to 1.0.
     */
    types::FeatureMatrix predict(const types::FeatureMatrix& data, Proportions) const;

    /**
     * @brief Add a tree to the forest.
     *
     * @param tree  Tree to add (ownership transferred).
     */
    void add_tree(std::unique_ptr<BootstrapTree> tree);

    bool operator==(const Forest& other) const;
    bool operator!=(const Forest& other) const;

    void accept(ModelVisitor& visitor) const override;

    /**
     * @brief Out-of-bag predictions by majority vote.
     *
     * For each observation, predicts using majority vote of only the
     * trees where it was out-of-bag.  Observations with no OOB tree
     * receive a sentinel value (−1) and are excluded when computing
     * oob_error().
     *
     * @param x  Training feature matrix (n × p).
     * @return   OOB predictions (n); −1 where no OOB tree exists.
     */
    types::ResponseVector oob_predict(const types::FeatureMatrix& x) const;

    /**
     * @brief Out-of-bag error rate.
     *
     * For each observation, predicts using majority vote of only the
     * trees where it was out-of-bag, then computes the overall
     * misclassification rate.
     *
     * @param x  Training feature matrix (n × p).
     * @param y  Training response vector (n).
     * @return   Error rate in [0, 1], or −1 if no observation has any
     *           OOB tree.
     */
    double oob_error(
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y) const;
  };
}
