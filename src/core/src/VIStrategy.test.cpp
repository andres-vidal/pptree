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


  const int n_vars = 2;
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.16962195932865143,
    0.26724648475646973,
    0.00253944098949432,
    0.36136019229888916,
    0.14448873698711395;

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

  const int n_vars = data.cols();
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
    0.497305,
    0.00889968,
    0.0137289,
    0.0177429,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566;

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


  const int n_vars = 2;
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorAdjustedStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.19476625323295593,
    0.18999017775058746,
    0.00231075938791036,
    0.17675185203552246,
    0.06259537488222122;


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

  const int n_vars = data.cols();
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIProjectorAdjustedStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
    0.983637,
    0.018022,
    0.028957,
    0.036786,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617;

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
    0.33333,
    0.44444,
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


  const int n_vars = 2;
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIPermutationStrategy<float, int>()(forest);

  DVector<float> expected(5);
  expected <<
    0.34999999403953552,
    0.20000000298023224,
    0,
    0,
    0;


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

  const int n_vars = data.cols();
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = VIPermutationStrategy<float, int>()(forest);

  DVector<float> expected(12);
  expected <<
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
    0.0,
    0.0;

  ASSERT_APPROX(expected, result);
}
