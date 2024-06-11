#include <gtest/gtest.h>

#include "Projector.hpp"

using namespace models::pp;
using namespace models::stats;

TEST(Projector, ProjectDataZeroProjector) {
  Data<long double> data(30, 5);
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
    2, 8, 0, 0, 1,
    2, 8, 0, 0, 2,
    2, 8, 1, 0, 2,
    2, 8, 1, 0, 1,
    2, 8, 0, 1, 1,
    2, 8, 0, 1, 2,
    2, 8, 2, 1, 1,
    2, 8, 1, 1, 1,
    2, 8, 1, 1, 2,
    2, 8, 2, 1, 2,
    2, 8, 1, 2, 1,
    2, 8, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  Projector<long double> projector({ 0.0, 0.0, 0.0, 0.0, 0.0 });

  DataColumn<long double> actual = projector.project(data);
  DataColumn<long double> expected = DataColumn<long double>::Zero(30);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Projector, ProjectDataGeneric) {
  Data<long double> data(30, 5);
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
    2, 8, 0, 0, 1,
    2, 8, 0, 0, 2,
    2, 8, 1, 0, 2,
    2, 8, 1, 0, 1,
    2, 8, 0, 1, 1,
    2, 8, 0, 1, 2,
    2, 8, 2, 1, 1,
    2, 8, 1, 1, 1,
    2, 8, 1, 1, 2,
    2, 8, 2, 1, 2,
    2, 8, 1, 2, 1,
    2, 8, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  Projector<long double> projector({ -0.02965,  0.08452, -0.24243, -0.40089, -0.87892 });

  DataColumn<long double> actual = projector.project(data);
  DataColumn<long double> expected(30);
  expected <<
    -1.30946,
    -0.27208,
    -0.90857,
    -1.55189,
    -1.30946,
    -0.67297,
    -1.30946,
    -2.43081,
    -0.83143,
    -0.91540,
    -0.26206,
    -1.14098,
    -1.38341,
    -0.50449,
    -0.66295,
    -1.54187,
    -1.14781,
    -0.90538,
    -1.78430,
    -2.02673,
    -1.30627,
    -1.14781,
    -0.46961,
    -1.34853,
    -1.59096,
    -0.71204,
    -0.87050,
    -1.74942,
    -1.35536,
    -1.11293;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_TRUE(expected.isApprox(actual, 0.00001));
}

TEST(Projector, PProjectDataColumnZeroProjector) {
  DataColumn<long double> data(5);
  data <<
    1.0, 2.0, 3.0, 4.0, 5.0;

  Projector<long double> projector({ 0.0, 0.0, 0.0, 0.0, 0.0 });

  long double result = projector.project(data);

  ASSERT_EQ(0, result);
}

TEST(Projector, PProjectDataColumnGeneric) {
  DataColumn<long double> data(5);
  data <<
    1.0, 2.0, 3.0, 4.0, 5.0;

  Projector<long double> projector({ -0.02965,  0.08452, -0.24243, -0.40089, -0.87892 });

  long double result = projector.project(data);

  ASSERT_NEAR(-6.58606, result, 0.00001);
}
