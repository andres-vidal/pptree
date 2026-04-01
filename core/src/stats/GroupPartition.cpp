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

  GroupPartition::BlockMap GroupPartition::init_Blocks(GroupVector const& y) {
    std::map<Group, Block> Blocks;

    for (int i = 0; i < y.rows(); i++) {
      if (Blocks.count(y(i)) == 0) {
        Group curr   = y(i);
        Blocks[curr] = Block{.start = i};

        if (i != 0) {
          Group prev        = y(i - 1);
          Blocks[curr].prev = prev;
          Blocks[prev].next = curr;
          Blocks[prev].end  = i - 1;
          Blocks[prev].size = i - Blocks[prev].start;
        }
      } else if (y(i - 1) != y(i)) {
        throw std::invalid_argument("GroupPartition: data is not organized in contiguous groups");
      }
    }

    Group last           = y(y.rows() - 1);
    Blocks.at(last).end  = y.rows() - 1;
    Blocks.at(last).size = y.rows() - Blocks.at(last).start;

    return Blocks;
  }

  GroupPartition::GroupMap GroupPartition::init_supergroups() {
    GroupMap sg;

    for (Group const& g : groups) {
      sg[g] = g;
    }

    return sg;
  }

  GroupPartition::GroupPartition(GroupVector const& y)
      : groups(unique(y))
      , Blocks(init_Blocks(y))
      , supergroups(init_supergroups())
      , subgroups(utils::invert(supergroups)) {}

  GroupPartition::GroupPartition(BlockMap const& Blocks_, GroupSet const& groups_)
      : groups(groups_)
      , Blocks(Blocks_)
      , supergroups(init_supergroups())
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
    FeatureVector global_mean = mean(x);
    FeatureMatrix result      = FeatureMatrix::Zero(x.cols(), x.cols());

    GroupSet groups = utils::values(supergroups);

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

    GroupSet groups = utils::values(supergroups);

    for (Group const& g : groups) {
      auto group_data = group(x, g);
      auto centered   = group_data.rowwise() - group_data.colwise().mean();

      result += centered.transpose() * centered;
    }

    return result;
  }

  GroupPartition GroupPartition::subset(GroupSet groups_) const {
    BlockMap subset_Blocks;
    std::optional<Group> prev;

    for (auto const& g : groups_) {
      Block Block = {.start = Blocks.at(g).start, .end = Blocks.at(g).end, .size = Blocks.at(g).size};

      if (prev) {
        Block.prev                   = *prev;
        subset_Blocks.at(*prev).next = g;
      }

      subset_Blocks[g] = Block;
      prev             = g;
    }

    return GroupPartition(subset_Blocks, groups_);
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
