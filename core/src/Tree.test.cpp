#include <gtest/gtest.h>

#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"
#include "VIStrategy.hpp"

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
  Data<float> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;




  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  Data<float> data(15, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3;

  DataColumn<int> groups(15, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse<float, int>::make(0),
      TreeCondition<float, int>::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse<float, int>::make(1),
        TreeResponse<float, int>::make(2))),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1, 2 })));


  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  Data<float> data(10, 4);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)
      ),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateThreeGroups) {
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      TreeCondition<float, int>::make(
        as_projector({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)),
      TreeResponse<float, int>::make(2)),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1, 2 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfUnivariateTwoGroups) {
  Data<float> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfMultivariateTwoGroups) {
  Data<float> data(10, 12);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> result = Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)
      ),
    TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));


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

TEST(Tree, RetrainLDASameDataSpec) {
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> result = tree.retrain(SortedDataSpec<float, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainLDADifferentDataSpec) {
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Data<float> other_data(10, 4);
  other_data <<
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

  DataColumn<int> other_groups(10);
  other_groups <<
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

  Tree<float, int> result = tree.retrain(SortedDataSpec<float, int>(other_data, other_groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)
      ),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, RetrainPDASameDataSpec) {
  Data<float> data(10, 12);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups));

  Tree<float, int> result = tree.retrain(SortedDataSpec<float, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainPDADifferentDataSpec) {
  Data<float> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups));

  Data<float> other_data(10, 1);
  other_data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> other_groups(10, 1);
  other_groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<float, int> result = tree.retrain(SortedDataSpec<float, int>(other_data, other_groups));

  Tree<float, int> expect = Tree<float, int>(
    TreeCondition<float, int>::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse<float, int>::make(0),
      TreeResponse<float, int>::make(1)),
    TrainingSpec<float, int>::lda(),
    SortedDataSpec<float, int>(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, VariableImportanceProjectorLDAMultivariateThreeGroups) {
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(*TrainingSpec<float, int>::lda(),
      SortedDataSpec<float, int>(data, groups));

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
  Data<float> data(10, 12);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups));


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
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(*TrainingSpec<float, int>::lda(),
      SortedDataSpec<float, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportanceProjectorAdjustedPDAMultivariateTwoGroups) {
  Data<float> data(10, 12);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups));


  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationLDAMultivariateThreeGroups) {
  Data<float> data(30, 5);
  data <<
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

  DataColumn<int> groups(30);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(*TrainingSpec<float, int>::lda(),
      SortedDataSpec<float, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationPDAMultivariateTwoGroups) {
  Data<float> data(10, 12);
  data <<
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

  DataColumn<int> groups(10);
  groups <<
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

  Tree<float, int> tree = Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(0.5),
    SortedDataSpec<float, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<float, int>()), std::invalid_argument);
}

TEST(Tree, ErrorRateDataSpecMin) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  float result = tree.error_rate(SortedDataSpec<float, int>(x, actual_y));

  ASSERT_FLOAT_EQ(0.0, result);
}

TEST(Tree, ErrorRateDataSpecMax) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Constant(30, 3);

  float result = tree.error_rate(SortedDataSpec<float, int>(x, actual_y));

  ASSERT_FLOAT_EQ(1.0, result);
}

TEST(Tree, ErrorRateDataSpecGeneric) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Zero(30);

  float result = tree.error_rate(SortedDataSpec<float, int>(x, actual_y));

  ASSERT_NEAR(0.666, result, 0.1);
}

TEST(Tree, ErrorRateBootstrapDataSpecMin) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = tree.error_rate(BootstrapDataSpec<float, int>(x, actual_y, sample_indices));

  ASSERT_FLOAT_EQ(0.0, result);
}

TEST(Tree, ErrorRateBootstrapDataSpecMax) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Constant(30, 1);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = tree.error_rate(BootstrapDataSpec<float, int>(x, actual_y, sample_indices));

  ASSERT_FLOAT_EQ(1.0, result);
}

TEST(Tree, ErrorRateBootstrapDataSpecGeneric) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Zero(30);

  std::vector<int> sample_indices(20);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = tree.error_rate(BootstrapDataSpec<float, int>(x, actual_y, sample_indices));


  ASSERT_NEAR(0.5, result, 0.1);
}

TEST(Tree, ConfusionMatrixDataSpecDiagonal) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  ConfusionMatrix result = tree.confusion_matrix(SortedDataSpec<float, int>(x, actual_y));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 10, 12, 8;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Tree, ConfusionMatrixDataSpecZeroDiagonal) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
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

  ConfusionMatrix result = tree.confusion_matrix(SortedDataSpec<float, int>(x, actual_y));

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

TEST(Tree, ConfusionMatrixBootstrapDataSpecDiagonal) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree    = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  ConfusionMatrix result = tree.confusion_matrix(BootstrapDataSpec<float, int>(x, actual_y, sample_indices));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 4, 4, 4;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Tree, ConfusionMatrixBootstrapDataSpecZeroDiagonal) {
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

  SortedDataSpec<float, int> data(x, y);
  Tree<float, int> tree = Tree<float, int>::train(*TrainingSpec<float, int>::lda(), data);
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

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  ConfusionMatrix result = tree.confusion_matrix(BootstrapDataSpec<float, int>(x, actual_y, sample_indices));

  Data<int> expected(3, 3);
  expected <<
    0, 0, 4,
    4, 0, 0,
    0, 4, 0;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}
