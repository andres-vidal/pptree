#include <gtest/gtest.h>

#include "Tree.hpp"

using namespace models;
using namespace models::pp;
using namespace models::stats;

static Projector<long double> as_projector(std::vector<long double> vector) {
  Eigen::Map<Projector<long double> > projector(vector.data(), vector.size());
  return projector;
}

TEST(ResponseEquals, true_case) {
  Response<long double, int> r1(1);
  Response<long double, int> r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(ResponseEquals, false_case) {
  Response<long double, int> r1(1);
  Response<long double, int> r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(ConditionEquals, true_case) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 1.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 2.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_approximate_thresholds) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.000000000000001,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, false_case_non_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 2.0, 3.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_thresholds) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 1.0, 2.0 }),
    4.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_responses) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_structures) {
  Condition<long double, int> c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeEquals, true_case) {
  Tree<long double, int> t1(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))));

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(TreeEquals, false_case) {
  Tree<long double, int> t1(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(2)));

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(PPTreeTrainLDA, univariate_two_groups) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;




  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, univariate_three_groups) {
  Data<long double> data(15, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3;

  DataColumn<int> groups(15, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 })));


  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, multivariate_two_groups) {
  Data<long double> data(10, 4);
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

  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)
      ),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, multivariate_three_groups) {
  Data<long double> data(30, 5);
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

  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainPDA, lambda_onehalf_univariate_two_groups) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainPDA, lambda_onehalf_multivariate_two_groups) {
  Data<long double> data(10, 12);
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

  Tree<long double, int> result = train(
    TrainingSpec<long double, int>::glda(0.5),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)
      ),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::glda(0.5)),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));


  ASSERT_EQ(expect, result);
}

TEST(PPTreePredictDataColumn, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)));


  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))));

  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictDataColumn, multivariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)));

  DataColumn<long double> input(4);
  input << 1, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 4, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, multivariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)));

  DataColumn<long double> input(5);
  input << 1, 0, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2, 5, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);

  input << 9, 8, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictData, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)));

  Data<long double> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))));

  Data<long double> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, multivariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)));

  Data<long double> input(2, 4);
  input <<
    1, 0, 1, 1,
    4, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, multivariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)));

  Data<long double> input(3, 5);
  input <<
    1, 0, 0, 1, 1,
    2, 5, 0, 0, 1,
    9, 8, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}

TEST(PPTreeLDARetrain, idempotent_in_same_data_spec) {
  Data<long double> data(30, 5);
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

  Tree<long double, int> tree = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> result = tree.retrain(DataSpec<long double, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(PPTreeLDARetrain, generates_a_different_tree_with_different_data_spec) {
  Data<long double> data(30, 5);
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

  Tree<long double, int> tree = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Data<long double> other_data(10, 4);
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

  Tree<long double, int> result = tree.retrain(DataSpec<long double, int>(other_data, other_groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)
      ),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreePDARetrain, idempotent_in_same_data_spec) {
  Data<long double> data(10, 12);
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

  Tree<long double, int> tree = train(
    TrainingSpec<long double, int>::glda(0.5),
    DataSpec<long double, int>(data, groups));

  Tree<long double, int> result = tree.retrain(DataSpec<long double, int>(data, groups));

  ASSERT_EQ(tree, result);
}

TEST(PPTreePDARetrain, generates_a_different_tree_with_different_data_spec) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> tree = train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Data<long double> other_data(10, 1);
  other_data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> other_groups(10, 1);
  other_groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> result = tree.retrain(DataSpec<long double, int>(other_data, other_groups));

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    std::make_unique<TrainingSpec<long double, int> >(TrainingSpec<long double, int>::lda()),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1 })));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeLDAVariableImportance, multivariate_three_groups) {
  Data<long double> data(30, 5);
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

  Tree<long double, int> tree = train(TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));

  Projector<long double> result = tree.variable_importance();

  Projector<long double> expected = as_projector(
    {   0.408057,
        0.553833,
        0.00341304,
        0.00643757,
        0.0160685 });

  ASSERT_TRUE(expected.isApprox(result, 0.0001));
}

TEST(PPTreePDAVariableImportante, multivariate_two_groups) {
  Data<long double> data(10, 12);
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

  Tree<long double, int> tree = train(
    TrainingSpec<long double, int>::glda(0.5),
    DataSpec<long double, int>(data, groups));


  Projector<long double> result = tree.variable_importance();

  Projector<long double> expected = as_projector({
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
