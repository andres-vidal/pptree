#pragma once

#include "Invariant.hpp"

#define APPROX_THRESHOLD 0.01

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, APPROX_THRESHOLD)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(models::math::collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b


#define ASSERT_EQ_DATA(a, b) \
        ASSERT_EQ(a.size(), b.size()); \
        ASSERT_EQ(a.rows(), b.rows()); \
        ASSERT_EQ(a.cols(), b.cols()); \
        ASSERT_EQ(a, b);

// Workaround for GCC miscompilation of long overloaded operator, chains
// (see https://stackoverflow.com/questions/79872387).
// Use these instead of Eigen's comma initializer (operator<<) for matrices and vectors.

#define DATA(T, ROWS, ...)                                         \
        ([&]() -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {                     \
          const int _rows = static_cast<int>(ROWS);                                      \
          std::vector<T> _vals = { __VA_ARGS__ };                                        \
          invariant(_rows > 0, "ROWS must be > 0");         \
          invariant(_vals.size() % _rows == 0, "Element count not divisible by ROWS");          \
          const int _cols = static_cast<int>(_vals.size()) / _rows;                       \
          return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(_vals.data(), _rows, _cols); \
        })()
