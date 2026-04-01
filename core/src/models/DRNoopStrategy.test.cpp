#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/DRNoopStrategy.hpp"
#include "models/DRStrategy.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::dr;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(DRNoopStrategy, FromJsonValid) {
  json j        = {{"name", "noop"}};
  auto strategy = DRNoopStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(DRNoopStrategy, FromJsonRoundTrip) {
  json j        = {{"name", "noop"}};
  auto strategy = DRNoopStrategy::from_json(j);

  json out;
  strategy->to_json(out);

  EXPECT_EQ(out["name"], "noop");
  EXPECT_EQ(out.size(), 1u);
}

TEST(DRNoopStrategy, FromJsonUnknownParam) {
  json j = {{"name", "noop"}, {"unexpected", 0}};
  EXPECT_THROW(DRNoopStrategy::from_json(j), std::runtime_error);
}

TEST(DRNoopStrategy, RegistryLookup) {
  json j        = {{"name", "noop"}};
  auto strategy = DRStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);

  json out;
  strategy->to_json(out);
  EXPECT_EQ(out["name"], "noop");
}

TEST(DRNoopStrategy, SelectsAllColumns) {
  FeatureMatrix x = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

  ResponseVector y = VEC(Response, 0, 1, 1);
  GroupPartition gp(y);
  RNG rng(0);

  DRNoopStrategy dr;
  auto result = dr.select(x, gp, rng);

  ASSERT_EQ(result.selected_cols.size(), 4u);
  EXPECT_EQ(result.original_size, 4);
  EXPECT_EQ(result.selected_cols, (std::vector<int>{0, 1, 2, 3}));
}
