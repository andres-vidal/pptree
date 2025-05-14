#include <gtest/gtest.h>

#include "Tree.hpp"

#include "TrainingSpec.hpp"
#include "TrainingSpecGLDA.hpp"
#include "TrainingSpecUGLDA.hpp"

#include "Macros.hpp"

using namespace models;
using namespace models::pp;
using namespace models::stats;
using namespace models::math;

static Projector<float> as_projector(std::vector<float> vector) {
  Eigen::Map<Projector<float> > projector(vector.data(), vector.size());
  return projector;
}

TEST(TreeResponse, EqualsEqualResponses) {
  TreeResponse<float, int> r1(1);
  TreeResponse<float, int> r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(TreeResponse, EqualsDifferentResponses) {
  TreeResponse<float, int> r1(1);
  TreeResponse<float, int> r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(TreeCondition, EqualsEqualConditions) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsCollinearProjectors) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 1.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 2.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsApproximateThresholds) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.000000000000001,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsNonCollinearProjectors) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 0.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 0.0, 1.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentThresholds) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 1.0, 2.0 }),
    4.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentResponses) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentStructures) {
  TreeCondition<float, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeResponse<float, int>::make(2));

  TreeCondition<float, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse<float, int>::make(1),
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse<float, int>::make(1),
      TreeResponse<float, int>::make(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(Tree, EqualsEqualTrees) {
  Tree<float, int> t1(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse<float, int>::make(1),
      TreeCondition<float, int>::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))));

  Tree<float, int> t2(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse<float, int>::make(1),
      TreeCondition<float, int>::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  Tree<float, int> t1(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse<float, int>::make(1),
      TreeResponse<float, int>::make(2)));

  Tree<float, int> t2(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse<float, int>::make(1),
      TreeResponse<float, int>::make(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  Data<float> x(10, 1);
  x <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> y(10, 1);
  y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;


  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  Data<float> x(15, 1);
  x <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3;

  DataColumn<int> y(15, 1);
  y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse<float, int>::make(0),
      TreeCondition<float, int>::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))));


  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  Data<float> x(10, 4);
  x <<
    1, 0, 1, 1,
    1, 1, 0, 0,
    1, 0, 0, 1,
    1, 1, 1, 1,
    4, 0, 0, 1,
    4, 0, 0, 2,
    4, 0, 0, 3,
    4, 1, 0, 1,
    4, 0, 1, 1,
    4, 0, 1, 2;

  DataColumn<int> y(10);
  y <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateThreeGroups) {
  Data<float> x(30, 5);
  x <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> y(30);
  y <<
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
    2;

  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      TreeCondition<float, int>::make(
        as_projector({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)),
      TreeResponse<float, int>::make(2)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfUnivariateTwoGroups) {
  Data<float> x(10, 1);
  x <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> y(10, 1);
  y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfMultivariateTwoGroups) {
  Data<float> x(10, 12);
  x <<
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2;

  DataColumn<int> y(10);
  y <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  stats::RNG rng(0);

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y, rng);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)
      ));


  ASSERT_EQ(expect, result);
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));


  DataColumn<float> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse<float, int>::make(0),
      TreeCondition<float, int>::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))));

  DataColumn<float> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  DataColumn<float> input(4);
  input << 1, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 4, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)),
      TreeResponse<float, int>::make(2)));

  DataColumn<float> input(5);
  input << 1, 0, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2, 5, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);

  input << 9, 8, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  Data<float> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse<float, int>::make(0),
      TreeCondition<float, int>::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))));

  Data<float> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  Data<float> input(2, 4);
  input <<
    1, 0, 1, 1,
    4, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  Tree<float, int> tree = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)),
      TreeResponse<float, int>::make(2)));

  Data<float> input(3, 5);
  input <<
    1, 0, 0, 1, 1,
    2, 5, 0, 0, 1,
    9, 8, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}
