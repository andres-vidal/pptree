#include "pptree.hpp"
#include "pptreeio.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;

TEST(ResponseEquals, true_case) {
  Response<double, int> r1(1);
  Response<double, int> r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(ResponseEquals, false_case) {
  Response<double, int> r1(1);
  Response<double, int> r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(ConditionEquals, true_case) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_collinear_projectors) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 1.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 2.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_approximate_thresholds) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 1.0, 2.0 }),
    3.000000000000001,
    new Response<double, int>(1),
    new Response<double, int>(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, false_case_non_collinear_projectors) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 2.0, 3.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_thresholds) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 1.0, 2.0 }),
    4.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_responses) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_structures) {
  Condition<double, int> c1(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Response<double, int>(2));

  Condition<double, int> c2(
    as_projector<double>({ 1.0, 2.0 }),
    3.0,
    new Response<double, int>(1),
    new Condition<double, int>(
      as_projector<double>({ 1.0, 2.0 }),
      3.0,
      new Response<double, int>(1),
      new Response<double, int>(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeEquals, true_case) {
  Tree<double, int> t1(
    new Condition<double, int>(
      as_projector<double>({ 1.0, 2.0 }),
      3.0,
      new Response<double, int>(1),
      new Condition<double, int>(
        as_projector<double>({ 1.0, 2.0 }),
        3.0,
        new Response<double, int>(1),
        new Response<double, int>(2))));

  Tree<double, int> t2(
    new Condition<double, int>(
      as_projector<double>({ 1.0, 2.0 }),
      3.0,
      new Response<double, int>(1),
      new Condition<double, int>(
        as_projector<double>({ 1.0, 2.0 }),
        3.0,
        new Response<double, int>(1),
        new Response<double, int>(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(TreeEquals, false_case) {
  Tree<double, int> t1(
    new Condition<double, int>(
      as_projector<double>({ 1.0, 2.0 }),
      3.0,
      new Response<double, int>(1),
      new Response<double, int>(2)));

  Tree<double, int> t2(
    new Condition<double, int>(
      as_projector<double>({ 1.0, 2.0 }),
      3.0,
      new Response<double, int>(1),
      new Response<double, int>(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(PPTreeTrain, lda_strategy_univariate_two_groups) {
  Data<double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<double, int> result = pptree::train(
    data,
    groups,
    (PPStrategy<double, int>)lda_strategy<double, int>);

  Tree<double, int> expect = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_univariate_three_groups) {
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

  Tree<double, int> result = pptree::train(
    data,
    groups,
    (PPStrategy<double, int>)lda_strategy<double, int>);

  Tree<double, int> expect = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.75,
      new Response<double, int>(0),
      new Condition<double, int>(
        as_projector<double>({ 1.0 }),
        2.5,
        new Response<double, int>(1),
        new Response<double, int>(2))));


  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_multivariate_two_groups) {
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

  Tree<double, int> result = train(
    data,
    groups,
    (PPStrategy<double, int>)lda_strategy<double, int>);

  Tree<double, int> expect = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ -1.0, 1.1437956e-16, 1.3801543e-16, 1.9571836e-16 }),
      -2.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrain, lda_strategy_multivariate_three_groups) {
  Data<double> data(30, 5);
  data <<
    1, 0, 0, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 1, 1,
    1, 0, 0, 1, 1,
    1, 0, 1, 1, 0,
    1, 0, 0, 1, 1,
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

  Tree<double, int> result = train(
    data,
    groups,
    (PPStrategy<double, int>)lda_strategy<double, int>);

  Tree<double, int> expect = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ -0.9805807, 0.1961161, -1.7850380e-16, -2.1664469e-16, -1.18058054e-15 }),
      -4.1184388379018655,
      new Response<double, int>(2),
      new Condition<double, int>(
        as_projector<double>({ 0.0906722, 0.0868016, -2.98384873e-17, 1.2586933e-17, -8.0883903e-17 }),
        0.3530121908270415,
        new Response<double, int>(0),
        new Response<double, int>(1))));


  ASSERT_EQ(expect, result);
}

TEST(PPTreePredictDataColumn, univariate_two_groups) {
  Tree<double, int> tree = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));


  DataColumn<double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, univariate_three_groups) {
  Tree<double, int> tree = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.75,
      new Response<double, int>(0),
      new Condition<double, int>(
        as_projector<double>({ 1.0 }),
        2.5,
        new Response<double, int>(1),
        new Response<double, int>(2))));

  DataColumn<double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictData, univariate_two_groups) {
  Tree<double, int> tree = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));

  Data<double> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, univariate_three_groups) {
  Tree<double, int> tree = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.75,
      new Response<double, int>(0),
      new Condition<double, int>(
        as_projector<double>({ 1.0 }),
        2.5,
        new Response<double, int>(1),
        new Response<double, int>(2))));

  Data<double> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}
