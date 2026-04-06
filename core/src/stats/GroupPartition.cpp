#include "stats/GroupPartition.hpp"

#include <stdexcept>

using namespace ppforest2::types;

namespace ppforest2::stats {
  bool GroupPartition::is_contiguous(GroupVector const& y) {
    GroupSet visited;

    for (int i = 0; i < y.rows(); i++) {
      if (visited.count(y(i)) == 0) {
        visited.insert(y(i));
      } else if (y(i - 1) != y(i)) {
        return false;
      }
    }

    return true;
  }

  GroupPartition::BlockMap GroupPartition::init_blocks(GroupVector const& y) {
    std::map<Group, Block> blocks;

    for (int i = 0; i < y.rows(); i++) {
      if (blocks.count(y(i)) == 0) {
        Group curr   = y(i);
        blocks[curr] = Block{.start = i};

        if (i != 0) {
          Group prev        = y(i - 1);
          blocks[curr].prev = prev;
          blocks[prev].next = curr;
          blocks[prev].end  = i - 1;
          blocks[prev].size = i - blocks[prev].start;
        }
      } else if (y(i - 1) != y(i)) {
        throw std::invalid_argument("GroupPartition: data is not organized in contiguous groups");
      }
    }

    Group const last     = y(y.rows() - 1);
    blocks.at(last).end  = y.rows() - 1;
    blocks.at(last).size = y.rows() - blocks.at(last).start;

    return blocks;
  }

  GroupPartition::GroupMap GroupPartition::init_supergroups(GroupSet const& groups) {
    GroupMap sg;

    for (Group const& g : groups) {
      sg[g] = g;
    }

    return sg;
  }

  GroupPartition::GroupPartition(GroupVector const& y)
      : groups(unique(y))
      , Blocks(init_blocks(y))
      , supergroups(init_supergroups(groups))
      , subgroups(utils::invert(supergroups)) {}

  GroupPartition::GroupPartition(BlockMap const& Blocks_)
      : groups(utils::keys(Blocks_))
      , Blocks(Blocks_)
      , supergroups(init_supergroups(groups))
      , subgroups(utils::invert(supergroups)) {}

  GroupPartition::GroupPartition(BlockMap const& Blocks_, GroupSet const& groups_)
      : groups(groups_)
      , Blocks(Blocks_)
      , supergroups(init_supergroups(groups_))
      , subgroups(utils::invert(supergroups)) {}

  GroupPartition::GroupPartition(BlockMap const& Blocks_, GroupMap const& supergroups_)
      : groups(utils::values(supergroups_))
      , Blocks(Blocks_)
      , supergroups(supergroups_)
      , subgroups(utils::invert(supergroups)) {}

  GroupPartition::GroupPartition(BlockMap const& Blocks_, GroupSet const& groups_, GroupMap const& supergroups_)
      : groups(groups_)
      , Blocks(Blocks_)
      , supergroups(supergroups_)
      , subgroups(utils::invert(supergroups)) {}

  int GroupPartition::group_start(Group const& group) const {
    return Blocks.at(group).start;
  }

  int GroupPartition::group_end(Group const& group) const {
    return Blocks.at(group).end;
  }

  int GroupPartition::group_size(Group const& group) const {
    return Blocks.at(group).size;
  }

  FeatureVector GroupPartition::mean(FeatureMatrix const& x) const {
    return data(x).colwise().mean();
  }

  FeatureMatrix GroupPartition::bgss(FeatureMatrix const& x) const {
    FeatureVector const global_mean = mean(x);
    FeatureMatrix result            = FeatureMatrix::Zero(x.cols(), x.cols());

    GroupSet const groups = utils::values(supergroups);

    for (Group const& g : groups) {
      auto group_data = group(x, g);
      auto group_mean = group_data.colwise().mean().transpose();
      auto centered   = group_mean - global_mean;

      result += group_data.rows() * (centered * centered.transpose());
    }

    return result;
  }

  FeatureMatrix GroupPartition::wgss(FeatureMatrix const& x) const {
    FeatureMatrix result = FeatureMatrix::Zero(x.cols(), x.cols());

    GroupSet const groups = utils::values(supergroups);

    for (Group const& g : groups) {
      auto group_data = group(x, g);
      auto centered   = group_data.rowwise() - group_data.colwise().mean();

      result += centered.transpose() * centered;
    }

    return result;
  }

  GroupPartition GroupPartition::subset(GroupSet const& groups) const {
    BlockMap subset_blocks;
    std::optional<Group> prev;

    for (auto const& g : groups) {
      Block Block = {.start = Blocks.at(g).start, .end = Blocks.at(g).end, .size = Blocks.at(g).size};

      if (prev) {
        Block.prev                   = *prev;
        subset_blocks.at(*prev).next = g;
      }

      subset_blocks[g] = Block;
      prev             = g;
    }

    return GroupPartition(subset_blocks, groups);
  }

  std::pair<GroupPartition, GroupPartition> GroupPartition::split(SplitSizes const& left_sizes) const {
    BlockMap l_blocks;
    BlockMap r_blocks;

    for (auto const& [g, block] : Blocks) {
      int const left_count = left_sizes.count(g) != 0U ? left_sizes.at(g) : 0;

      invariant(left_count >= 0 && left_count <= block.size, "GroupPartition::split: left_count out of range");

      if (left_count > 0) {
        l_blocks.emplace(g, Block{block.start, block.start + left_count - 1, left_count});
      }

      if (left_count < block.size) {
        r_blocks.emplace(g, Block{block.start + left_count, block.end, block.size - left_count});
      }
    }

    return {GroupPartition(l_blocks), GroupPartition(r_blocks)};
  }

  GroupPartition GroupPartition::remap(GroupMap const& mapping) const {
    return GroupPartition(Blocks, mapping);
  }

  GroupPartition GroupPartition::collapse() const {
    GroupMap mapping;
    for (auto const& g : groups) {
      mapping[g] = 0;
    }

    return remap(mapping);
  }
}
