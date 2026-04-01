#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/SRMeanOfMeansStrategy.hpp"
#include "models/SRStrategy.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::sr;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(SRMeanOfMeansStrategy, FromJsonValid) {
  json j        = {{"name", "mean_of_means"}};
  auto strategy = SRMeanOfMeansStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(SRMeanOfMeansStrategy, FromJsonRoundTrip) {
  json j        = {{"name", "mean_of_means"}};
  auto strategy = SRMeanOfMeansStrategy::from_json(j);

  json out;
  strategy->to_json(out);

  EXPECT_EQ(out["name"], "mean_of_means");
  EXPECT_EQ(out.size(), 1u);
}

TEST(SRMeanOfMeansStrategy, FromJsonUnknownParam) {
  json j = {{"name", "mean_of_means"}, {"extra", 1}};
  EXPECT_THROW(SRMeanOfMeansStrategy::from_json(j), std::runtime_error);
}

TEST(SRMeanOfMeansStrategy, RegistryLookup) {
  json j        = {{"name", "mean_of_means"}};
  auto strategy = SRStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);

  json out;
  strategy->to_json(out);
  EXPECT_EQ(out["name"], "mean_of_means");
}

TEST(SRMeanOfMeansStrategy, RegistryUnknownStrategy) {
  json j = {{"name", "unknown_sr"}};
  EXPECT_THROW(SRStrategy::from_json(j), std::runtime_error);
}

TEST(SRMeanOfMeansStrategy, ThresholdIsMidpointOfProjectedMeans) {
  // group_1: rows [1,2] and [3,4], mean = [2, 3]
  // group_2: rows [5,6] and [7,8], mean = [6, 7]
  // projector = [1, 0] (identity on first column)
  // projected mean_1 = 2, projected mean_2 = 6
  // threshold = (2 + 6) / 2 = 4
  FeatureMatrix g1 = MAT(Feature, rows(2), 1, 2, 3, 4);

  FeatureMatrix g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  pp::Projector proj = VEC(Feature, 1, 0);

  SRMeanOfMeansStrategy sr;
  Feature t = sr.threshold(g1, g2, proj);

  EXPECT_FLOAT_EQ(t, 4.0f);
}

TEST(SRMeanOfMeansStrategy, ThresholdWithNonTrivialProjector) {
  // group_1: [1,0], [0,1] → mean = [0.5, 0.5]
  // group_2: [4,0], [0,4] → mean = [2, 2]
  // projector = [1, 1]
  // projected mean_1 = 0.5 + 0.5 = 1.0
  // projected mean_2 = 2 + 2 = 4.0
  // threshold = (1 + 4) / 2 = 2.5
  FeatureMatrix g1 = MAT(Feature, rows(2), 1, 0, 0, 1);

  FeatureMatrix g2 = MAT(Feature, rows(2), 4, 0, 0, 4);

  pp::Projector proj = VEC(Feature, 1, 1);

  SRMeanOfMeansStrategy sr;
  EXPECT_FLOAT_EQ(sr.threshold(g1, g2, proj), 2.5f);
}

TEST(SRMeanOfMeansStrategy, ThresholdSymmetric) {
  // Swapping groups should give the same threshold
  FeatureMatrix g1 = MAT(Feature, rows(2), 1, 2, 3, 4);

  FeatureMatrix g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  pp::Projector proj = VEC(Feature, 0, 1);

  SRMeanOfMeansStrategy sr;
  EXPECT_FLOAT_EQ(sr.threshold(g1, g2, proj), sr.threshold(g2, g1, proj));
}

TEST(SRMeanOfMeansStrategy, OperatorCallEqualsThreshold) {
  FeatureMatrix g1 = MAT(Feature, rows(2), 1, 2, 3, 4);

  FeatureMatrix g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  pp::Projector proj = VEC(Feature, 1, 0);

  SRMeanOfMeansStrategy sr;
  EXPECT_FLOAT_EQ(sr(g1, g2, proj), sr.threshold(g1, g2, proj));
}
