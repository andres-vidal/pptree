#pragma once

#include "models/Model.hpp"
#include "models/BootstrapTree.hpp"

#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace ppforest2 {
  /**
   * @brief A projection pursuit random forest.
   *
   * An ensemble of BootstrapTree instances, each trained on a
   * bootstrap sample.  Predictions are made by majority vote.
   * Out-of-bag estimation and vote-proportion predictions are
   * supported.
   *
   * @code
   *   TrainingSpec spec(pp::pda(0.0), vars::uniform(3), cutpoint::mean_of_means(), 500, 0);
   *   Forest forest = Forest::train(spec, x, y);
   *
   *   types::OutcomeVector preds = forest.predict(x_test);
   *   double oob = forest.oob_error(x, y);
   *
   *   // Vote proportions — (n × G) matrix, rows sum to 1.
   *   types::FeatureMatrix probs = forest.predict(x_test, Proportions{});
   * @endcode
   */
  struct Forest : public Model {
    /**
     * @brief Train a random forest.
     *
     * Forest-level parameters (size, seed, threads, max_retries)
     * are read from the training specification.
     *
     * @param training_spec  Training specification.
     * @param x              Feature matrix (n × p).
     * @param y              Outcome vector (n).
     * @return               Trained forest.
     */
    static Forest
    train(TrainingSpec const& training_spec, types::FeatureMatrix const& x, types::OutcomeVector const& y);

    std::vector<BootstrapTree::Ptr> trees;

    Forest();
    explicit Forest(TrainingSpec::Ptr training_spec);

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Prediction.
     */
    types::Outcome predict(types::FeatureVector const& data) const override;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predictions (n).
     */
    types::OutcomeVector predict(types::FeatureMatrix const& data) const override;

    /**
     * @brief Predict vote proportions for a matrix of observations.
     *
     * For each observation, counts votes from every tree and returns
     * the proportion of trees that voted for each group.  The number
     * of groups G is derived from the root node of the first tree.
     *
     * @param data  Feature matrix (n × p).
     * @return      Vote proportions matrix (n × G), rows sum to 1.0.
     */
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    /**
     * @brief Add a tree to the forest.
     *
     * @param tree  Tree to add (ownership transferred).
     */
    void add_tree(std::unique_ptr<BootstrapTree> tree);

    bool operator==(Forest const& other) const;
    bool operator!=(Forest const& other) const;

    void accept(Model::Visitor& visitor) const override;

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
    types::OutcomeVector oob_predict(types::FeatureMatrix const& x) const;

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
    double oob_error(types::FeatureMatrix const& x, types::OutcomeVector const& y) const;
  };
}
