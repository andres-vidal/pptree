#include "pptree.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;

TEST(PPTreeTrain, lda_strategy_unidimensional_data) {
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

  ASSERT_EQ(result.root.threshold, 1.5);
  ASSERT_EQ(result.root.left->response, 0);
  ASSERT_EQ(result.root.right->response, 1);
}
