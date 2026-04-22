#include "stats/GroupPartition.hpp"

#include "utils/Invariant.hpp"

#include <cmath>
#include <stdexcept>

using namespace ppforest2::types;

namespace ppforest2::stats {
  namespace {
    // Convert a float response to integer group labels, but only if every
    // entry is a finite, integer-valued float. Guards against
    // `ByLabel::init(OutcomeVector)` or `GroupPartition(OutcomeVector)`
    // being handed a truly continuous response (e.g. a regression spec
    // miswired with `grouping_by_label`), where `.cast<GroupId>()` would
    // silently truncate to useless labels.
    GroupIdVector validate_and_cast_to_ids(OutcomeVector const& y) {
      for (Eigen::Index i = 0; i < y.size(); ++i) {
        Outcome const v = y(i);
        if (!std::isfinite(v) || v != std::floor(v)) {
          invariant(
              false,
              "GroupPartition(OutcomeVector): y[" + std::to_string(i) + "] = " + std::to_string(v) +
                  " is not an integer value. "
                  "Classification labels must be integer-encoded; if this is a "
                  "continuous response, use `grouping::by_cutpoint()` with "
                  "regression mode instead."
          );
        }
      }
      return y.cast<GroupId>();
    }
  }

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

  GroupPartition::GroupPartition(types::OutcomeVector const& y)
      : GroupPartition(validate_and_cast_to_ids(y)) {}

  bool GroupPartition::is_contiguous(types::OutcomeVector const& y) {
    types::GroupIdVector const y_int = y.cast<GroupId>();
    return is_contiguous(y_int);
  }

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

  int GroupPartition::total_size() const {
    int total = 0;
    for (auto const& kv : Blocks) {
      total += kv.second.size;
    }
    return total;
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

  GroupPartition GroupPartition::two_groups(int start0, int end0, int start1, int end1) {
    invariant(start0 >= 0 && end0 >= start0, "GroupPartition::two_groups: invalid group 0 range");
    invariant(start1 >= 0 && end1 >= start1, "GroupPartition::two_groups: invalid group 1 range");
    invariant(start1 == end0 + 1, "GroupPartition::two_groups: groups must be adjacent");

    int size0 = end0 - start0 + 1;
    int size1 = end1 - start1 + 1;

    BlockMap blocks;
    blocks[0] = Block{start0, end0, size0, 1, std::nullopt};
    blocks[1] = Block{start1, end1, size1, std::nullopt, 0};

    return GroupPartition(blocks);
  }

  GroupPartition GroupPartition::single_group(int start, int end) {
    invariant(start >= 0 && end >= start, "GroupPartition::single_group: invalid range");

    int size = end - start + 1;

    BlockMap blocks;
    blocks[0] = Block{start, end, size, std::nullopt, std::nullopt};

    return GroupPartition(blocks);
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
