#include <gtest/gtest.h>

#include "VIStrategy.hpp"

#include "Macros.hpp"

using namespace models;
using namespace models::stats;
using namespace models::math;

TEST(VIProjectorStrategy, TreeLDAMultivariateThreeGroups) {
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

  DVector<float> result = VIProjectorStrategy<float, int>()(tree);

  DataColumn<float> expected(5);
  expected <<
    0.408057,
    0.553833,
    0.00341304,
    0.00643757,
    0.0160685;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorStrategy, TreePDAMultivariateTwoGroups) {
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


  DataColumn<float> result = VIProjectorStrategy<float, int>()(tree);

  DataColumn<float> expected(12);
  expected <<
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
    0.00180949;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorStrategy, BootstrapTreeLDAMultivariateThreeGroups) {
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

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::lda(), data);

  DVector<float> result = VIProjectorStrategy<float, int>()(tree);

  DataColumn<float> expected(5);
  expected <<
    0.327572,
    0.561704,
    0.0,
    0.0,
    0.0;


  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorStrategy, BootstrapTreePDAMultivariateTwoGroups) {
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

  std::vector<int> sample_indices = { 0, 2, 6, 8 };

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::glda(0.1), data);

  DataColumn<float> result = VIProjectorStrategy<float, int>()(tree);

  DataColumn<float> expected(12);
  expected <<
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorStrategy, ForestLDASomeVariablesMultivariateThreeGroups) {
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


  const int n_vars   = 2;
  const float lambda = 0;
  const int seed     = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.29120835661888123,
    0.37499451637268066,
    0.04791311174631118,
    0.09619309753179550,
    0.07325347512960434;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorStrategy, ForestPDAAllVariablesMultivariateTwoGroups) {
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

  const int n_vars   = data.cols();
  const float lambda = 0.1;
  const int seed     = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
    0.4973304271697998,
    0.0065120994113385,
    0.0112658599391579,
    0.0023133084177970,
    0.0105210691690444,
    0.0105212237685918,
    0.0105212181806564,
    0.0105212191119790,
    0.0105212163180112,
    0.0105212163180112,
    0.0105212200433015,
    0.0105212181806564;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorAdjustedStrategy, TreeLDAMultivariateThreeGroups) {
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


  auto strategy = VIProjectorAdjustedStrategy<float, int>();

  ASSERT_THROW(strategy(tree), std::invalid_argument);
}

TEST(VIProjectorAdjustedStrategy, TreePDAMultivariateTwoGroups) {
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


  auto strategy = VIProjectorAdjustedStrategy<float, int>();

  ASSERT_THROW(strategy(tree), std::invalid_argument);
}

TEST(VIProjectorAdjustedStrategy, BootstrapLDATreeMultivariateThreeGroups) {
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

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::lda(), data);

  DVector<float> result = VIProjectorAdjustedStrategy<float, int>()(tree);

  DataColumn<float> expected(5);
  expected <<
    0.491359,
    0.592556,
    0.0,
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorAdjustedStrategy, BootstrapPDATreeMultivariateTwoGroups) {
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

  std::vector<int> sample_indices = { 0, 2, 6, 8 };

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::glda(0.1), data);

  DataColumn<float> result = VIProjectorAdjustedStrategy<float, int>()(tree);

  DataColumn<float> expected(12);
  expected <<
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorAdjustedStrategy, ForestLDASomeVariablesMultivariateThreeGroups) {
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


  const int n_vars   = 2;
  const float lambda = 0;
  const int seed     = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorAdjustedStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.31015947461128235,
    0.33594012260437012,
    0.01743866316974163,
    0.04736451059579849,
    0.02686723135411739;


  ASSERT_APPROX(expected, result);
}

TEST(VIProjectorAdjustedStrategy, ForestPDAAllVariablesMultivariateTwoGroups) {
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

  const int n_vars   = data.cols();
  const float lambda = 0.1;
  const int seed     = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorAdjustedStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
    0.98541688919067383,
    0.01270248740911483,
    0.02189479023218154,
    0.00450026756152510,
    0.02072355896234512,
    0.02072386071085929,
    0.02072384767234325,
    0.02072384953498840,
    0.02072384580969810,
    0.02072384394705295,
    0.02072385139763355,
    0.02072384953498840;

  ASSERT_APPROX(expected, result);
}

TEST(VIPermutationStrategy, TreeLDAMultivariateThreeGroups) {
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


  auto strategy = VIPermutationStrategy<float, int>();

  ASSERT_THROW(strategy(tree), std::invalid_argument);
}

TEST(VIPermutationStrategy, TreePDAMultivariateTwoGroups) {
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


  auto strategy = VIPermutationStrategy<float, int>();

  ASSERT_THROW(strategy(tree), std::invalid_argument);
}

TEST(VIPermutationStrategy, BootstrapTreeLDAMultivariateThreeGroups) {
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

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  Random::seed(0);

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::lda(), data);

  DVector<float> result = VIPermutationStrategy<float, int>()(tree);

  DataColumn<float> expected(5);
  expected <<
    0.44444,
    0.27777,
    0.00000,
    0.00000,
    0.00000;

  ASSERT_APPROX(expected, result);
}

TEST(VIPermutationStrategy, BootstrapTreePDAMultivariateTwoGroups) {
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

  std::vector<int> sample_indices = { 0, 2, 6, 8 };

  BootstrapDataSpec<float, int> data(x, y, sample_indices);
  BootstrapTree<float, int> tree = BootstrapTree<float, int>::train(*TrainingSpec<float, int>::glda(0.1), data);

  Random::seed(0);

  DataColumn<float> result = VIPermutationStrategy<float, int>()(tree);

  DataColumn<float> expected(12);
  expected <<
    0.333337,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}

TEST(VIPermutationStrategy, ForestLDASomeVariablesMultivariateThreeGroups) {
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


  const int n_vars   = 2;
  const float lambda = 0;
  const int seed     = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIPermutationStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.07083333283662796,
    0.22499999403953552,
    -0.0041666701436042786,
    0,
    0.024999991059303284;

  ASSERT_APPROX(expected, result);
}

TEST(VIPermutationStrategy, ForestPDAAllVariablesMultivariateTwoGroups) {
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

  const int n_vars   = data.cols();
  const float lambda = 0.1;
  const int seed     = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIPermutationStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
    0.22499999403953552,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}
