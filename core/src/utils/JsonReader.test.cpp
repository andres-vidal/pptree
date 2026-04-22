#include <gtest/gtest.h>

#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using json = nlohmann::json;

namespace {
  testing::AssertionResult throws_with(std::function<void()> const& fn, std::string const& needle) {
    try {
      fn();
    } catch (std::runtime_error const& e) {
      std::string const what = e.what();
      if (what.find(needle) != std::string::npos) {
        return testing::AssertionSuccess();
      }
      return testing::AssertionFailure() << "expected message containing \"" << needle << "\", got \"" << what << "\"";
    } catch (std::exception const& e) {
      return testing::AssertionFailure() << "threw std::exception (not runtime_error): " << e.what();
    }
    return testing::AssertionFailure() << "did not throw";
  }
}

TEST(JsonReader, RequireObjectOnNonObjectThrowsWithPath) {
  json const j = 42;
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with([&] { r.require_object(); }, "config: expected object"));
}

TEST(JsonReader, RequireMissingKeyNamesPath) {
  json const j = {{"name", "pda"}};
  JsonReader const r(j, "pp");
  EXPECT_TRUE(throws_with([&] { r.require<float>("lambda"); }, "pp.lambda: missing required key"));
}

TEST(JsonReader, RequireWrongTypeNamesPath) {
  json const j = {{"name", "pda"}, {"lambda", "not a number"}};
  JsonReader const r(j, "pp");
  EXPECT_TRUE(throws_with([&] { r.require<float>("lambda"); }, "pp.lambda: unexpected type"));
}

TEST(JsonReader, RequireTypedValue) {
  json const j = {{"name", "pda"}, {"lambda", 0.5}};
  JsonReader const r(j, "pp");
  EXPECT_EQ(r.require<std::string>("name"), "pda");
  EXPECT_FLOAT_EQ(r.require<float>("lambda"), 0.5F);
}

TEST(JsonReader, OptionalMissingReturnsFallback) {
  json const j = {{"name", "pda"}};
  JsonReader const r(j, "pp");
  EXPECT_FLOAT_EQ(r.optional<float>("lambda", 0.0F), 0.0F);
}

TEST(JsonReader, OptionalPresentReturnsValue) {
  json const j = {{"name", "pda"}, {"lambda", 0.5}};
  JsonReader const r(j, "pp");
  EXPECT_FLOAT_EQ(r.optional<float>("lambda", 0.0F), 0.5F);
}

TEST(JsonReader, RequireEnumValid) {
  json const j = {{"mode", "regression"}};
  JsonReader const r(j, "config");
  EXPECT_EQ(r.require_enum("mode", {"classification", "regression"}), "regression");
}

TEST(JsonReader, RequireEnumInvalid) {
  json const j = {{"mode", "regresion"}};
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with(
      [&] { (void)r.require_enum("mode", {"classification", "regression"}); },
      "config.mode: must be one of [classification, regression] (got 'regresion')"
  ));
}

TEST(JsonReader, RequireEnumWrongType) {
  json const j = {{"mode", 1}};
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with([&] { (void)r.require_enum("mode", {"a", "b"}); }, "config.mode: expected string"));
}

TEST(JsonReader, RequireIntRangeOk) {
  json const j = {{"size", 5}};
  JsonReader const r(j, "config");
  EXPECT_EQ(r.require_int("size", 0), 5);
}

TEST(JsonReader, RequireIntOutOfRange) {
  json const j = {{"size", -1}};
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with([&] { (void)r.require_int("size", 0); }, "config.size: must be in [0, ∞]"));
}

TEST(JsonReader, RequireIntRejectsNonIntegerFloat) {
  json const j = {{"size", 5.5}};
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with([&] { (void)r.require_int("size"); }, "config.size: expected integer"));
}

TEST(JsonReader, RequireIntAcceptsIntegerValuedFloat) {
  // R has no distinct JSON integer — everything serialized as number. A
  // value like `5.0` is semantically an integer and should round-trip.
  json const j = {{"size", 5.0}};
  JsonReader const r(j, "config");
  EXPECT_EQ(r.require_int("size"), 5);
}

TEST(JsonReader, RequireIntRejectsFloatsOutsideLongLongRange) {
  // `static_cast<long long>(d)` is UB when `d` isn't representable in
  // `long long`. The guard catches values at/beyond +/- 2^63 before the
  // cast is attempted.
  json const j = {{"k", 1e20}};
  JsonReader const r(j, "config");
  EXPECT_TRUE(throws_with([&] { (void)r.require_int("k"); }, "config.k: integer out of representable range"));
}

TEST(JsonReader, RequireNumberRangeOk) {
  json const j = {{"lambda", 0.5}};
  JsonReader const r(j, "pp");
  EXPECT_DOUBLE_EQ(r.require_number("lambda", 0.0, 1.0), 0.5);
}

TEST(JsonReader, RequireNumberOutOfRange) {
  json const j = {{"lambda", 1.5}};
  JsonReader const r(j, "pp");
  EXPECT_TRUE(throws_with([&] { (void)r.require_number("lambda", 0.0, 1.0); }, "pp.lambda: must be in [0, 1]"));
}

TEST(JsonReader, OnlyKeysAcceptsSubset) {
  json const j = {{"name", "pda"}};
  JsonReader const r(j, "pp");
  EXPECT_NO_THROW(r.only_keys({"name", "lambda"}));
}

TEST(JsonReader, OnlyKeysRejectsUnknown) {
  json const j = {{"name", "pda"}, {"lamda", 0.5}};
  JsonReader const r(j, "pp");
  EXPECT_TRUE(
      throws_with([&] { r.only_keys({"name", "lambda"}); }, "pp: unexpected key 'lamda' (allowed: name, lambda)")
  );
}

TEST(JsonReader, AtDescendsAndCarriesPath) {
  json const j = {{"config", {{"mode", "regression"}}}};
  JsonReader const r(j, "Export");
  auto const config = r.at("config");
  EXPECT_EQ(config.path(), "Export.config");
  EXPECT_EQ(config.require_enum("mode", {"classification", "regression"}), "regression");
}

TEST(JsonReader, AtOnMissingKeyNamesPath) {
  json const j = json::object();
  JsonReader const r(j, "Export");
  EXPECT_TRUE(throws_with([&] { (void)r.at("config"); }, "Export.config: missing required key"));
}

TEST(JsonReader, AtOnNonObjectNamesPath) {
  json const j = {{"config", "oops"}};
  JsonReader const r(j, "Export");
  EXPECT_TRUE(throws_with([&] { (void)r.at("config"); }, "Export.config: expected object"));
}

TEST(JsonReader, RequireArrayReturnsRef) {
  json const j = {{"rules", json::array({1, 2, 3})}};
  JsonReader const r(j, "stop");
  auto const& arr = r.require_array("rules");
  EXPECT_EQ(arr.size(), 3u);
}

TEST(JsonReader, RequireArrayOnNonArrayThrows) {
  json const j = {{"rules", 7}};
  JsonReader const r(j, "stop");
  EXPECT_TRUE(throws_with([&] { (void)r.require_array("rules"); }, "stop.rules: expected array"));
}

TEST(JsonReader, EmptyPathOmitsLeadingDot) {
  json const j = {{"k", 1}};
  JsonReader const r(j, "");
  EXPECT_TRUE(throws_with([&] { (void)r.require<float>("missing"); }, "missing: missing"));
}
