#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "Tree.hpp"

#include "TrainingSpec.hpp"
#include "TrainingSpecGLDA.hpp"
#include "TrainingSpecUGLDA.hpp"

#include "Macros.hpp"

using namespace models;
using namespace models::pp;
using namespace models::stats;
using namespace models::types;
using namespace models::math;

static Projector as_projector(std::vector<Feature> vector) {
  Eigen::Map<Projector > projector(vector.data(), vector.size());
  return projector;
}

TEST(TreeResponse, EqualsEqualResponses) {
  TreeResponse r1(1);
  TreeResponse r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(TreeResponse, EqualsDifferentResponses) {
  TreeResponse r1(1);
  TreeResponse r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(TreeCondition, EqualsEqualConditions) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsCollinearProjectors) {
  TreeCondition c1(
    as_projector({ 1.0, 1.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 2.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsApproximateThresholds) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.000000000000001,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsNonCollinearProjectors) {
  TreeCondition c1(
    as_projector({ 1.0, 0.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 0.0, 1.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentThresholds) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    4.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentResponses) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentStructures) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(Tree, EqualsEqualTrees) {
  Tree t1(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeCondition::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  Tree t2(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeCondition::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  Tree t1(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(2)));

  Tree t2(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  FeatureMatrix x = DATA(Feature, 10,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2);

  ResponseVector y = DATA(Response, 10,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1);


  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.0), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  FeatureMatrix x = DATA(Feature, 15,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2,
      3, 3, 3, 3, 3);

  ResponseVector y = DATA(Response, 15,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.0), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2))));


  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  FeatureMatrix x = DATA(Feature, 10,
      1, 0, 1, 1,
      1, 1, 0, 0,
      1, 0, 0, 1,
      1, 1, 1, 1,
      4, 0, 0, 1,
      4, 0, 0, 2,
      4, 0, 0, 3,
      4, 1, 0, 1,
      4, 0, 1, 1,
      4, 0, 1, 2);

  ResponseVector y = DATA(Response, 10,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.0), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateThreeGroups) {
  FeatureMatrix x = DATA(Feature, 30,
      1, 0, 1, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 2, 1,
      1, 0, 0, 1, 1,
      1, 1, 1, 1, 0,
      1, 0, 0, 2, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 5, 0, 0, 1,
      2, 5, 0, 0, 2,
      3, 5, 1, 0, 2,
      2, 5, 1, 0, 1,
      2, 5, 0, 1, 1,
      2, 5, 0, 1, 2,
      2, 5, 2, 1, 1,
      2, 5, 1, 1, 1,
      2, 5, 1, 1, 2,
      2, 5, 2, 1, 2,
      2, 5, 1, 2, 1,
      2, 5, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = DATA(Response, 30,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.0), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      TreeCondition::make(
        as_projector({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        TreeResponse::make(0),
        TreeResponse::make(1)),
      TreeResponse::make(2)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfUnivariateTwoGroups) {
  FeatureMatrix x = DATA(Feature, 10,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2);

  ResponseVector y = DATA(Response, 10,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.0), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfMultivariateTwoGroups) {
  FeatureMatrix x = DATA(Feature, 10,
      1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);

  ResponseVector y = DATA(Response, 10,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecGLDA(0.5), x, y, rng);

  Tree expect = Tree(
    TreeCondition::make(
      as_projector({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      TreeResponse::make(0),
      TreeResponse::make(1)
      ));


  ASSERT_EQ(expect, result);
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));


  FeatureVector input = DATA(Feature, 1, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = DATA(Feature, 1, 2.0);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  FeatureVector input = DATA(Feature, 1, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = DATA(Feature, 1, 2.0);
  ASSERT_EQ(tree.predict(input), 1);

  input = DATA(Feature, 1, 3.0);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureVector input = DATA(Feature, 4, 1, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = DATA(Feature, 4, 4, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse::make(0),
        TreeResponse::make(1)),
      TreeResponse::make(2)));

  FeatureVector input = DATA(Feature, 5, 1, 0, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = DATA(Feature, 5, 2, 5, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);

  input = DATA(Feature, 5, 9, 8, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureMatrix input = DATA(Feature, 2, 1.0, 2.0);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = DATA(Response, 2, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  FeatureMatrix input = DATA(Feature, 3, 1.0, 2.0, 3.0);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = DATA(Response, 3, 0, 1, 2);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureMatrix input = DATA(Feature, 2,
      1, 0, 1, 1,
      4, 0, 0, 1);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = DATA(Response, 2, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse::make(0),
        TreeResponse::make(1)),
      TreeResponse::make(2)));

  FeatureMatrix input = DATA(Feature, 3,
      1, 0, 0, 1, 1,
      2, 5, 0, 0, 1,
      9, 8, 0, 0, 1);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = DATA(Response, 3, 0, 1, 2);

  ASSERT_EQ(result, expected);
}
