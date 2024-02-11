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
