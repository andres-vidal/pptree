#include "pptree.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;

TEST(PPTreeResponseToString, returns_json) {
  Response<double, int> response(1);
  ASSERT_EQ(response.to_string(), "{\"value\":1}");
}

TEST(PPTreeConditionToString, returns_json) {
  Projector<double> projector(2);
  projector << 1, 2;

  Condition<double, int> condition(
    projector,
    1.5,
    new Response<double, int>(0),
    new Response<double, int>(1));

  ASSERT_EQ(
    condition.to_string(),
    "{\"projector\":[1,2],\"threshold\":1.5,\"lower\":{\"value\":0},\"upper\":{\"value\":1}}");
}

TEST(PPTreeTreeToString, returns_json) {
  Projector<double> projector(2);
  projector << 1, 2;

  Condition<double, int> *condition = new Condition<double, int>(
    projector,
    1.5,
    new Response<double, int>(0),
    new Response<double, int>(1));

  Tree<double, int> tree(condition);

  ASSERT_EQ(
    tree.to_string(),
    "{\"root\":{\"projector\":[1,2],\"threshold\":1.5,\"lower\":{\"value\":0},\"upper\":{\"value\":1}}}");
}

TEST(PPTreeTrain, lda_strategy_unidimensional_data_two_groups) {
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

  Tree<double, int> expected = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.5,
      new Response<double, int>(0),
      new Response<double, int>(1)));

  ASSERT_STREQ(result.to_string().c_str(), expected.to_string().c_str());
}

TEST(PPTreeTrain, lda_strategy_unidimensional_data_three_groups) {
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

  Tree<double, int> expected = Tree<double, int>(
    new Condition<double, int>(
      as_projector<double>({ 1.0 }),
      1.75,
      new Response<double, int>(0),
      new Condition<double, int>(
        as_projector<double>({ 1.0 }),
        2.5,
        new Response<double, int>(1),
        new Response<double, int>(2))));


  ASSERT_STREQ(result.to_string().c_str(), expected.to_string().c_str());
}
