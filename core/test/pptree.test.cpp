#include "pptree.hpp"
#include "pptreeio.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;


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

  json json_result(result);
  json json_expect(expect);

  ASSERT_STREQ(json_expect.dump().c_str(), json_result.dump().c_str());
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


  json json_result(result);
  json json_expect(expect);

  ASSERT_STREQ(json_expect.dump().c_str(), json_result.dump().c_str());
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
      as_projector<double>({ -1.0, 1.1437956370464563e-16, 1.380154307539727e-16, 1.957183583530815e-16 }),
      -2.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));

  json json_result(result);
  json json_expect(expect);

  ASSERT_STREQ(json_expect.dump().c_str(), json_result.dump().c_str());
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
      as_projector<double>({ -0.9805806756909201, 0.19611613513818413, -1.785038044022064e-16, -2.166446927983865e-16, -1.1805805462575629e-15 }),
      -4.1184388379018655,
      new Response<double, int>(2),
      new Condition<double, int>(
        as_projector<double>({ 0.09067218080194704, 0.08680156784964843, -2.983848690645762e-17, 1.2586932655869611e-17, -8.088390290592397e-17 }),
        0.3530121908270415,
        new Response<double, int>(0),
        new Response<double, int>(1))));

  json json_result(result);
  json json_expect(expect);

  ASSERT_STREQ(json_expect.dump().c_str(), json_result.dump().c_str());
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
