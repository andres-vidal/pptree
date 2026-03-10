/**
 * @file VarsSpec.test.cpp
 * @brief Tests for shared vars parsing (string and JSON overloads).
 */
#include <gtest/gtest.h>
#include "cli/VarsSpec.hpp"

using pptree::cli::parse_vars;
using json = nlohmann::json;

// --- String overload ---

TEST(VarsSpecString, DecimalProportion) {
  auto spec = parse_vars(std::string("0.5"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 0.5f);
}

TEST(VarsSpecString, DecimalProportionOne) {
  auto spec = parse_vars(std::string("1.0"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 1.0f);
}

TEST(VarsSpecString, SmallDecimal) {
  auto spec = parse_vars(std::string("0.01"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 0.01f);
}

TEST(VarsSpecString, Fraction) {
  auto spec = parse_vars(std::string("1/3"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_NEAR(spec.value, 1.0f / 3.0f, 1e-6);
}

TEST(VarsSpecString, FractionOne) {
  auto spec = parse_vars(std::string("3/3"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 1.0f);
}

TEST(VarsSpecString, IntegerCount) {
  auto spec = parse_vars(std::string("5"));

  EXPECT_FALSE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 5.0f);
}

TEST(VarsSpecString, IntegerCountOne) {
  auto spec = parse_vars(std::string("1"));

  EXPECT_FALSE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 1.0f);
}

TEST(VarsSpecString, DecimalZeroThrows) {
  EXPECT_THROW(parse_vars(std::string("0.0")), std::runtime_error);
}

TEST(VarsSpecString, DecimalAboveOneThrows) {
  EXPECT_THROW(parse_vars(std::string("1.5")), std::runtime_error);
}

TEST(VarsSpecString, NegativeDecimalThrows) {
  EXPECT_THROW(parse_vars(std::string("-0.5")), std::runtime_error);
}

TEST(VarsSpecString, NegativeIntegerThrows) {
  EXPECT_THROW(parse_vars(std::string("-1")), std::runtime_error);
}

TEST(VarsSpecString, ZeroIntegerThrows) {
  EXPECT_THROW(parse_vars(std::string("0")), std::runtime_error);
}

TEST(VarsSpecString, FractionAboveOneThrows) {
  EXPECT_THROW(parse_vars(std::string("3/2")), std::runtime_error);
}

TEST(VarsSpecString, FractionZeroDenominatorThrows) {
  EXPECT_THROW(parse_vars(std::string("1/0")), std::runtime_error);
}

TEST(VarsSpecString, FractionNegativeNumeratorThrows) {
  EXPECT_THROW(parse_vars(std::string("-1/3")), std::runtime_error);
}

TEST(VarsSpecString, FractionNegativeDenominatorThrows) {
  EXPECT_THROW(parse_vars(std::string("1/-3")), std::runtime_error);
}

// --- JSON overload ---

TEST(VarsSpecJson, FloatProportion) {
  auto spec = parse_vars(json(0.3));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 0.3f);
}

TEST(VarsSpecJson, FloatOne) {
  auto spec = parse_vars(json(1.0));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 1.0f);
}

TEST(VarsSpecJson, IntegerCount) {
  auto spec = parse_vars(json(5));

  EXPECT_FALSE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 5.0f);
}

TEST(VarsSpecJson, IntegerCountOne) {
  auto spec = parse_vars(json(1));

  EXPECT_FALSE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 1.0f);
}

TEST(VarsSpecJson, StringFraction) {
  auto spec = parse_vars(json("2/5"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 0.4f);
}

TEST(VarsSpecJson, StringDecimal) {
  auto spec = parse_vars(json("0.7"));

  EXPECT_TRUE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 0.7f);
}

TEST(VarsSpecJson, StringInteger) {
  auto spec = parse_vars(json("10"));

  EXPECT_FALSE(spec.is_proportion);
  EXPECT_FLOAT_EQ(spec.value, 10.0f);
}

TEST(VarsSpecJson, FloatAboveOneThrows) {
  EXPECT_THROW(parse_vars(json(1.5)), std::runtime_error);
}

TEST(VarsSpecJson, FloatZeroThrows) {
  EXPECT_THROW(parse_vars(json(0.0)), std::runtime_error);
}

TEST(VarsSpecJson, NegativeFloatThrows) {
  EXPECT_THROW(parse_vars(json(-0.5)), std::runtime_error);
}

TEST(VarsSpecJson, NegativeIntegerThrows) {
  EXPECT_THROW(parse_vars(json(-3)), std::runtime_error);
}

TEST(VarsSpecJson, ZeroIntegerThrows) {
  EXPECT_THROW(parse_vars(json(0)), std::runtime_error);
}
