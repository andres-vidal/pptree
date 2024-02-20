#include "pptree.hpp"
#include "pptreeio.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;

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
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 1.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 2.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_approximate_thresholds) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.000000000000001,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, false_case_non_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 2.0, 3.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_thresholds) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    4.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_responses) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_structures) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Response<long double, int>(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    new Response<long double, int>(1),
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      new Response<long double, int>(1),
      new Response<long double, int>(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeEquals, true_case) {
  Tree<long double, int> t1(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      new Response<long double, int>(1),
      new Condition<long double, int>(
        as_projector<long double>({ 1.0, 2.0 }),
        3.0,
        new Response<long double, int>(1),
        new Response<long double, int>(2))));

  Tree<long double, int> t2(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      new Response<long double, int>(1),
      new Condition<long double, int>(
        as_projector<long double>({ 1.0, 2.0 }),
        3.0,
        new Response<long double, int>(1),
        new Response<long double, int>(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(TreeEquals, false_case) {
  Tree<long double, int> t1(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      new Response<long double, int>(1),
      new Response<long double, int>(2)));

  Tree<long double, int> t2(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      new Response<long double, int>(1),
      new Response<long double, int>(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(PPTreeTrain, lda_strategy_univariate_two_groups) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> result = pptree::train(
    data,
    groups,
    (PPStrategy<long double, int>)lda_strategy<long double, int>);

  Tree<long double, int> expect = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.5,
      new Response<long double, int>(0),
      new Response<long double, int>(1)));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_univariate_three_groups) {
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

  Tree<long double, int> result = pptree::train(
    data,
    groups,
    (PPStrategy<long double, int>)lda_strategy<long double, int>);

  Tree<long double, int> expect = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.75,
      new Response<long double, int>(0),
      new Condition<long double, int>(
        as_projector<long double>({ 1.0 }),
        2.5,
        new Response<long double, int>(1),
        new Response<long double, int>(2))));


  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_multivariate_two_groups) {
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
    data,
    groups,
    (PPStrategy<long double, int>)lda_strategy<long double, int>);

  Tree<long double, int> expect = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      new Response<long double, int>(0),
      new Response<long double, int>(1)
      )
    );


  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_multivariate_three_groups) {
  Data<long double> data(30, 5);
  data <<
    1, 0, 1, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 2, 1,
    1, 0, 0, 1, 1,
    1, 0, 1, 1, 0,
    1, 0, 0, 2, 1,
    1, 0, 1, 1, 2,
    1, 0, 0, 2, 0,
    1, 0, 2, 1, 0,
    2, 5, 0, 0, 1,
    2, 5, 0, 0, 2,
    2, 5, 1, 0, 2,
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
    data,
    groups,
    (PPStrategy<long double, int>)lda_strategy<long double, int>);

  Tree<long double, int> expect = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      new Condition<long double, int>(
        as_projector<long double>({ 2.721658109136383e-17, -1.0, 0.0, 0.0, 0.0 }),
        -2.5,
        new Response<long double, int>(0),
        new Response<long double, int>(1)),
      new Response<long double, int>(2)));

  ASSERT_EQ(expect, result);
}

TEST(PPTreePredictDataColumn, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.5,
      new Response<long double, int>(0),
      new Response<long double, int>(1)));


  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.75,
      new Response<long double, int>(0),
      new Condition<long double, int>(
        as_projector<long double>({ 1.0 }),
        2.5,
        new Response<long double, int>(1),
        new Response<long double, int>(2))));

  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictData, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.5,
      new Response<long double, int>(0),
      new Response<long double, int>(1)));

  Data<long double> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    new Condition<long double, int>(
      as_projector<long double>({ 1.0 }),
      1.75,
      new Response<long double, int>(0),
      new Condition<long double, int>(
        as_projector<long double>({ 1.0 }),
        2.5,
        new Response<long double, int>(1),
        new Response<long double, int>(2))));

  Data<long double> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}
