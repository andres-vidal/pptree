#include "models/strategies/grouping/ByCutpoint.hpp"
#include "models/strategies/NodeContext.hpp"

#include <gtest/gtest.h>

using namespace ppforest2;
using namespace ppforest2::grouping;
using namespace ppforest2::stats;
using namespace ppforest2::types;

TEST(ByCutpoint, InitCreatesGroupPartition) {
  GroupIdVector y(6);
  y << 0, 0, 0, 1, 1, 1;

  ByCutpoint strategy;
  GroupPartition gp = strategy.init(y);

  EXPECT_EQ(gp.groups.size(), 2);
  EXPECT_EQ(gp.group_size(0), 3);
  EXPECT_EQ(gp.group_size(1), 3);
}

TEST(ByCutpoint, ToJson) {
  ByCutpoint strategy;
  auto j = strategy.to_json();

  EXPECT_EQ(j["name"], "by_cutpoint");
}

TEST(ByCutpoint, DisplayName) {
  ByCutpoint strategy;
  EXPECT_EQ(strategy.display_name(), "By cutpoint");
}

TEST(ByCutpoint, FromJson) {
  nlohmann::json j = {{"name", "by_cutpoint"}};
  auto ptr         = Grouping::from_json(j);

  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(ptr->display_name(), "By cutpoint");
}

TEST(ByCutpoint, FactoryFunction) {
  auto ptr = by_cutpoint();
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(ptr->display_name(), "By cutpoint");
}

TEST(GroupPartition, TwoGroupsFactory) {
  GroupPartition gp = GroupPartition::two_groups(0, 2, 3, 5);

  EXPECT_EQ(gp.groups.size(), 2);
  EXPECT_EQ(gp.group_start(0), 0);
  EXPECT_EQ(gp.group_end(0), 2);
  EXPECT_EQ(gp.group_size(0), 3);
  EXPECT_EQ(gp.group_start(1), 3);
  EXPECT_EQ(gp.group_end(1), 5);
  EXPECT_EQ(gp.group_size(1), 3);
}

TEST(GroupPartition, TwoGroupsAtOffset) {
  GroupPartition gp = GroupPartition::two_groups(10, 14, 15, 19);

  EXPECT_EQ(gp.groups.size(), 2);
  EXPECT_EQ(gp.group_start(0), 10);
  EXPECT_EQ(gp.group_end(0), 14);
  EXPECT_EQ(gp.group_size(0), 5);
  EXPECT_EQ(gp.group_start(1), 15);
  EXPECT_EQ(gp.group_end(1), 19);
  EXPECT_EQ(gp.group_size(1), 5);
}

TEST(GroupPartition, SingleGroupFactory) {
  GroupPartition gp = GroupPartition::single_group(5, 8);

  EXPECT_EQ(gp.groups.size(), 1);
  EXPECT_EQ(gp.group_start(0), 5);
  EXPECT_EQ(gp.group_end(0), 8);
  EXPECT_EQ(gp.group_size(0), 4);
}
