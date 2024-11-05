#include <gtest/gtest.h>

#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"
#include "VIStrategy.hpp"

using namespace models;
using namespace models::pp;
using namespace models::stats;
using namespace models::math;

static Projector<double> as_projector(std::vector<double> vector) {
  Eigen::Map<Projector<double> > projector(vector.data(), vector.size());
  return projector;
}

TEST(Response, EqualsEqualResponses) {
  Response<double, int> r1(1);
  Response<double, int> r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(Response, EqualsDifferentResponses) {
  Response<double, int> r1(1);
  Response<double, int> r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(Condition, EqualsEqualConditions) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(Condition, EqualsCollinearProjectors) {
  Condition<double, int> c1(
    as_projector({ 1.0, 1.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 2.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(Condition, EqualsApproximateThresholds) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.000000000000001,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(Condition, EqualsNonCollinearProjectors) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 2.0, 3.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(Condition, EqualsDifferentThresholds) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 1.0, 2.0 }),
    4.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(Condition, EqualsDifferentResponses) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(Condition, EqualsDifferentStructures) {
  Condition<double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Response<double, int> >(2));

  Condition<double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<double, int> >(1),
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<double, int> >(1),
      std::make_unique<Response<double, int> >(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(Tree, EqualsEqualTrees) {
  Tree<double, int> t1(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<double, int> >(1),
      std::make_unique<Condition<double, int> >(
        as_projector({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<double, int> >(1),
        std::make_unique<Response<double, int> >(2))));

  Tree<double, int> t2(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<double, int> >(1),
      std::make_unique<Condition<double, int> >(
        as_projector({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<double, int> >(1),
        std::make_unique<Response<double, int> >(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  Tree<double, int> t1(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<double, int> >(1),
      std::make_unique<Response<double, int> >(2)));

  Tree<double, int> t2(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<double, int> >(1),
      std::make_unique<Response<double, int> >(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  Data<double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;




  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  Data<double> data(15, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3;

  DataColumn<int> groups(15, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Condition<double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<double, int> >(1),
        std::make_unique<Response<double, int> >(2))),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1, 2 })));


  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  Data<double> data(10, 4);
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

  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)
      ),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainLDAMultivariateThreeGroups) {
  Data<double> data(30, 5);
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

  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      std::make_unique<Condition<double, int> >(
        as_projector({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        std::make_unique<Response<double, int> >(0),
        std::make_unique<Response<double, int> >(1)),
      std::make_unique<Response<double, int> >(2)),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1, 2 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfUnivariateTwoGroups) {
  Data<double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, TrainPDALambdaOnehalfMultivariateTwoGroups) {
  Data<double> data(10, 12);
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

  Tree<double, int> result = Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(0.5),
    DataSpec<double, int>(data, groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)
      ),
    TrainingSpec<double, int>::glda(0.5),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));


  ASSERT_EQ(expect, result);
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)));


  DataColumn<double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Condition<double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<double, int> >(1),
        std::make_unique<Response<double, int> >(2))));

  DataColumn<double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)));

  DataColumn<double> input(4);
  input << 1, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 4, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<double, int> >(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<double, int> >(0),
        std::make_unique<Response<double, int> >(1)),
      std::make_unique<Response<double, int> >(2)));

  DataColumn<double> input(5);
  input << 1, 0, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2, 5, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);

  input << 9, 8, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)));

  Data<double> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Condition<double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<double, int> >(1),
        std::make_unique<Response<double, int> >(2))));

  Data<double> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)));

  Data<double> input(2, 4);
  input <<
    1, 0, 1, 1,
    4, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  Tree<double, int> tree = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<double, int> >(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<double, int> >(0),
        std::make_unique<Response<double, int> >(1)),
      std::make_unique<Response<double, int> >(2)));

  Data<double> input(3, 5);
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
  Data<double> data(30, 5);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Tree<double, int> result = tree.retrain(DataSpec<double, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainLDADifferentDataSpec) {
  Data<double> data(30, 5);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Data<double> other_data(10, 4);
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

  Tree<double, int> result = tree.retrain(DataSpec<double, int>(other_data, other_groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)
      ),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, RetrainPDASameDataSpec) {
  Data<double> data(10, 12);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(0.5),
    DataSpec<double, int>(data, groups));

  Tree<double, int> result = tree.retrain(DataSpec<double, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(Tree, RetrainPDADifferentDataSpec) {
  Data<double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  Data<double> other_data(10, 1);
  other_data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> other_groups(10, 1);
  other_groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<double, int> result = tree.retrain(DataSpec<double, int>(other_data, other_groups));

  Tree<double, int> expect = Tree<double, int>(
    std::make_unique<Condition<double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<double, int> >(0),
      std::make_unique<Response<double, int> >(1)),
    TrainingSpec<double, int>::lda(),
    std::make_shared<DataSpec<double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(Tree, VariableImportanceProjectorLDAMultivariateThreeGroups) {
  Data<double> data(30, 5);
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

  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  DVector<double> result = tree.variable_importance(VIProjectorStrategy<double, int>());

  Projector<double> expected = as_projector(
    {   0.408057,
        0.553833,
        0.00341304,
        0.00643757,
        0.0160685 });

  ASSERT_TRUE(expected.isApprox(result, 0.0001));
}

TEST(Tree, VariableImportanceProjectorPDAMultivariateTwoGroups) {
  Data<double> data(10, 12);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(0.5),
    DataSpec<double, int>(data, groups));


  DVector<double> result = tree.variable_importance(VIProjectorStrategy<double, int>());

  Projector<double> expected = as_projector({
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

  ASSERT_TRUE(expected.isApprox(result, 0.0001));
}

TEST(Tree, VariableImportanceProjectorAdjustedLDAMultivariateThreeGroups) {
  Data<double> data(30, 5);
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

  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<double, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportanceProjectorAdjustedPDAMultivariateTwoGroups) {
  Data<double> data(10, 12);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(0.5),
    DataSpec<double, int>(data, groups));


  ASSERT_THROW(tree.variable_importance(VIProjectorAdjustedStrategy<double, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationLDAMultivariateThreeGroups) {
  Data<double> data(30, 5);
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

  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(),
    DataSpec<double, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<double, int>()), std::invalid_argument);
}

TEST(Tree, VariableImportancePermutationPDAMultivariateTwoGroups) {
  Data<double> data(10, 12);
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

  Tree<double, int> tree = Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(0.5),
    DataSpec<double, int>(data, groups));

  ASSERT_THROW(tree.variable_importance(VIPermutationStrategy<double, int>()), std::invalid_argument);
}

TEST(Tree, ErrorRateDataSpecMin) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  double result = tree.error_rate(DataSpec<double, int>(x, actual_y));

  ASSERT_DOUBLE_EQ(0.0, result);
}

TEST(Tree, ErrorRateDataSpecMax) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Constant(30, 3);

  double result = tree.error_rate(DataSpec<double, int>(x, actual_y));

  ASSERT_DOUBLE_EQ(1.0, result);
}

TEST(Tree, ErrorRateDataSpecGeneric) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Zero(30);

  double result = tree.error_rate(DataSpec<double, int>(x, actual_y));

  ASSERT_NEAR(0.666, result, 0.1);
}

TEST(Tree, ErrorRateBootstrapDataSpecMin) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  double result = tree.error_rate(BootstrapDataSpec<double, int>(x, actual_y, sample_indices));

  ASSERT_DOUBLE_EQ(0.0, result);
}

TEST(Tree, ErrorRateBootstrapDataSpecMax) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Constant(30, 1);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  double result = tree.error_rate(BootstrapDataSpec<double, int>(x, actual_y, sample_indices));

  ASSERT_DOUBLE_EQ(1.0, result);
}

TEST(Tree, ErrorRateBootstrapDataSpecGeneric) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = DataColumn<int>::Zero(30);

  std::vector<int> sample_indices(20);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  double result = tree.error_rate(BootstrapDataSpec<double, int>(x, actual_y, sample_indices));


  ASSERT_NEAR(0.5, result, 0.1);
}

TEST(Tree, ConfusionMatrixDataSpecDiagonal) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  ConfusionMatrix result = tree.confusion_matrix(DataSpec<double, int>(x, actual_y));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 10, 12, 8;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Tree, ConfusionMatrixDataSpecZeroDiagonal) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
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

  ConfusionMatrix result = tree.confusion_matrix(DataSpec<double, int>(x, actual_y));

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
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
  DataColumn<int> actual_y = tree.predict(data.x);

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  ConfusionMatrix result = tree.confusion_matrix(BootstrapDataSpec<double, int>(x, actual_y, sample_indices));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 4, 4, 4;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Tree, ConfusionMatrixBootstrapDataSpecZeroDiagonal) {
  Data<double> x(30, 5);
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

  DataSpec<double, int> data(x, y);
  Tree<double, int> tree = Tree<double, int>::train(*TrainingSpec<double, int>::lda(), data);
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

  ConfusionMatrix result = tree.confusion_matrix(BootstrapDataSpec<double, int>(x, actual_y, sample_indices));

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
