#include <gtest/gtest.h>

#include "Tree.hpp"
#include "VIStrategy.hpp"

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




  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

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

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

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

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

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

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

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

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

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

  Tree<float, int> result = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y);

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

TEST(Tree, RetrainLDASameGroupSpec) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  Tree<float, int> result = tree.retrain(x, y);

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainLDADifferentGroupSpec) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  Data<float> other_x(10, 4);
  other_x <<
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

  DataColumn<int> other_y(10);
  other_y <<
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

  Tree<float, int> result = tree.retrain(other_x, other_y);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)
      ));

  ASSERT_EQ(expect, result);
}

TEST(Tree, RetrainPDASameGroupSpec) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y);

  Tree<float, int> result = tree.retrain(x, y);

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainPDADifferentGroupSpec) {
  Data<float> x(10, 1);
  x <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> y(10, 1);
  y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  Data<float> other_x(10, 1);
  other_x <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> other_y(10, 1);
  other_y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<float, int> result = tree.retrain(other_x, other_y);

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)));

  ASSERT_EQ(expect, result);
}

TEST(Tree, VariableImportanceProjectorLDAMultivariateThreeGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  DVector<float> result = tree.variable_importance(VIProjectorStrategy<float, int>());

  Projector<float> expected = as_projector(
    {   0.408057,
        0.553833,
        0.00341304,
        0.00643757,
        0.0160685 });

  ASSERT_APPROX(expected, result);
}

TEST(Tree, VariableImportanceProjectorPDAMultivariateTwoGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y);


  DVector<float> result = tree.variable_importance(VIProjectorStrategy<float, int>());

  Projector<float> expected = as_projector({
    0.499665,
    0.00113766,
    0.00831906,
    0.0152932,
    0.00180949,
    0.00180949,
    0.00180949,
    0.00180949,
    0.00180949,
    0.00180949,
    0.00180949,
    0.00180949 });

  ASSERT_APPROX(expected, result);
}

TEST(Tree, VariableImportanceProjectorAdjustedLDAMultivariateThreeGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportanceProjectorAdjustedPDAMultivariateTwoGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y);


  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationLDAMultivariateThreeGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationPDAMultivariateTwoGroups) {
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

  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.5), x, y);

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, ErrorRateGroupSpecMin) {
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


  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  float result = tree.error_rate(x, tree.predict(x));

  ASSERT_FLOAT_EQ(0.0, result);
}

TEST(Tree, ErrorRateGroupSpecMax) {
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

  Tree<float, int> tree    = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);
  DataColumn<int> actual_y = DataColumn<int>::Constant(30, 3);

  float result = tree.error_rate(x, actual_y);

  ASSERT_FLOAT_EQ(1.0, result);
}

TEST(Tree, ErrorRateGroupSpecGeneric) {
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

  Tree<float, int> tree    = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);
  DataColumn<int> actual_y = DataColumn<int>::Zero(30);

  float result = tree.error_rate(x, actual_y);

  ASSERT_NEAR(0.666, result, 0.1);
}

TEST(Tree, ConfusionMatrixGroupSpecDiagonal) {
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

  Tree<float, int> tree    = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);
  DataColumn<int> actual_y = tree.predict(x);

  ConfusionMatrix result = tree.confusion_matrix(x, actual_y);

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 10, 12, 8;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Tree, ConfusionMatrixGroupSpecZeroDiagonal) {
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


  Tree<float, int> tree = Tree<float, int>::train(TrainingSpecGLDA<float, int>(0.0), x, y);

  DataColumn<int> actual_y(30);
  actual_y <<
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
    2,
    2,
    2,
    2,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0;

  ConfusionMatrix result = tree.confusion_matrix(x, actual_y);

  Data<int> expected(3, 3);
  expected <<
    0, 0, 8,
    10, 0, 0,
    0, 12, 0;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}
