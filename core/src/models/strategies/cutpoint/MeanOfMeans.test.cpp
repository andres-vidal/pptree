#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/cutpoint/MeanOfMeans.hpp"
#include "models/strategies/cutpoint/SplitCutpoint.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::cutpoint;
using namespace ppforest2::pp;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(CutpointMeanOfMeansStrategy, FromJsonValid) {
  json const j  = {{"name", "mean_of_means"}};
  auto strategy = MeanOfMeans::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(CutpointMeanOfMeansStrategy, FromJsonRoundTrip) {
  json const j  = {{"name", "mean_of_means"}};
  auto strategy = MeanOfMeans::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(CutpointMeanOfMeansStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "mean_of_means"}, {"extra", 1}};
  EXPECT_THROW(MeanOfMeans::from_json(j), std::runtime_error);
}

TEST(CutpointMeanOfMeansStrategy, RegistryLookup) {
  json const j  = {{"name", "mean_of_means"}};
  auto strategy = SplitCutpoint::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(CutpointMeanOfMeansStrategy, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_cutpoint"}};
  EXPECT_THROW(SplitCutpoint::from_json(j), std::runtime_error);
}

TEST(CutpointMeanOfMeansStrategy, CutpointIsMidpointOfProjectedMeans) {
  // group_1: rows [1,2] and [3,4], mean = [2, 3]
  // group_2: rows [5,6] and [7,8], mean = [6, 7]
  // projector = [1, 0] (identity on first column)
  // projected mean_1 = 2, projected mean_2 = 6
  // cutpoint = (2 + 6) / 2 = 4
  FeatureMatrix const g1 = MAT(Feature, rows(2), 1, 2, 3, 4);
  FeatureMatrix const g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  Projector const proj = VEC(Feature, 1, 0);

  MeanOfMeans const cp;
  Feature const t = cp.compute(g1, g2, proj);

  EXPECT_FLOAT_EQ(t, 4.0F);
}

TEST(CutpointMeanOfMeansStrategy, CutpointWithNonTrivialProjector) {
  // group_1: [1,0], [0,1] → mean = [0.5, 0.5]
  // group_2: [4,0], [0,4] → mean = [2, 2]
  // projector = [1, 1]
  // projected mean_1 = 0.5 + 0.5 = 1.0
  // projected mean_2 = 2 + 2 = 4.0
  // cutpoint = (1 + 4) / 2 = 2.5
  FeatureMatrix const g1 = MAT(Feature, rows(2), 1, 0, 0, 1);
  FeatureMatrix const g2 = MAT(Feature, rows(2), 4, 0, 0, 4);

  Projector const proj = VEC(Feature, 1, 1);

  MeanOfMeans const cp;
  EXPECT_FLOAT_EQ(cp.compute(g1, g2, proj), 2.5F);
}

TEST(CutpointMeanOfMeansStrategy, CutpointSymmetric) {
  // Swapping groups should give the same cutpoint
  FeatureMatrix const g1 = MAT(Feature, rows(2), 1, 2, 3, 4);
  FeatureMatrix const g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  Projector const proj = VEC(Feature, 0, 1);

  MeanOfMeans const cp;
  EXPECT_FLOAT_EQ(cp.compute(g1, g2, proj), cp.compute(g2, g1, proj));
}

TEST(CutpointMeanOfMeansStrategy, ComputeIsDeterministic) {
  FeatureMatrix const g1 = MAT(Feature, rows(2), 1, 2, 3, 4);
  FeatureMatrix const g2 = MAT(Feature, rows(2), 5, 6, 7, 8);

  Projector const proj = VEC(Feature, 1, 0);

  MeanOfMeans const cp;
  EXPECT_FLOAT_EQ(cp.compute(g1, g2, proj), cp.compute(g1, g2, proj));
}
