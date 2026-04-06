#pragma once


#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Map.hpp"
#include "utils/Invariant.hpp"

#include <map>
#include <optional>
#include <set>
#include <vector>
#include <Eigen/Dense>

namespace ppforest2::stats {
  /**
   * @brief Contiguous-block representation of grouped observations.
   *
   * Assumes the response vector is sorted so that observations of the
   * same group are contiguous.  Stores the start/end row indices of
   * each group block and provides efficient extraction, subsetting,
   * and computation of between- and within-group statistics.
   *
   * Groups can be hierarchically merged via remap(), which assigns
   * supergroup labels while tracking the original subgroups.
   *
   * @code
   *   // y must be sorted so equal values are contiguous.
   *   GroupPartition gp(y);
   *
   *   // Extract rows belonging to group 0:
   *   auto x_group0 = gp.group(x, 0);
   *
   *   // Between- and within-group statistics:
   *   auto B = gp.bgss(x);   // between-group sum of squares (p × p)
   *   auto W = gp.wgss(x);   // within-group sum of squares  (p × p)
   *
   *   // Restrict to a subset of groups:
   *   GroupPartition sub = gp.subset({0, 2});
   * @endcode
   */
  class GroupPartition {
    using Group       = types::Outcome;
    using GroupSet    = std::set<types::Outcome>;
    using GroupMap    = std::map<types::Outcome, types::Outcome>;
    using GroupInvMap = std::map<types::Outcome, GroupSet>;
    using GroupVector = types::Vector<types::Outcome>;

  public:
    /** @brief Check whether all equal values in @p y form a single contiguous block. */
    static bool is_contiguous(GroupVector const& y);

    /**
       * @brief Construct from a sorted response vector.
       *
       * @param y  Outcome vector (n) with contiguous group blocks.
       */
    GroupPartition(types::OutcomeVector const& y);

    /** @brief First row index of the block for @p group. */
    int group_start(Group const& group) const;
    /** @brief Last row index (inclusive) of the block for @p group. */
    int group_end(Group const& group) const;
    /** @brief Number of observations in @p group. */
    int group_size(Group const& group) const;

    /**
       * @brief Extract rows belonging to a group (or supergroup).
       *
       * Returns an Eigen block expression (zero-copy view) into @p x.
       * The result must be consumed immediately or assigned to a
       * concrete matrix — do not store it in `auto` across statements.
       *
       * @param x      Feature matrix (n × p).
       * @param group  Group label.
       * @return       Block expression over the rows of @p group.
       */
    template<typename Derived> auto group(Eigen::MatrixBase<Derived> const& x, Group const& group) const {
      std::vector<int> indices;

      auto const& subs = this->subgroups.at(group);

      for (auto const& g : subs) {
        for (int i = group_start(g); i <= group_end(g); ++i) {
          invariant(i >= 0 && i < x.rows(), "GroupPartition::group: index out of bounds");
          indices.push_back(i);
        }
      }

      return x(indices, Eigen::all);
    }

    /**
       * @brief Extract all rows across all groups.
       *
       * @param x  Feature matrix (n × p).
       * @return   Sub-matrix with all grouped rows.
       */
    template<typename Derived> auto data(Eigen::MatrixBase<Derived> const& x) const {
      std::vector<int> indices;

      for (auto const& kv : Blocks) {
        auto const& g = kv.first;
        for (int i = group_start(g); i <= group_end(g); ++i) {
          indices.push_back(i);
        }
      }

      return x(indices, Eigen::all);
    }

    /** @brief Overall mean of all grouped rows (p). */
    types::FeatureVector mean(types::FeatureMatrix const& x) const;
    /** @brief Between-group sum of squares matrix (p × p). */
    types::FeatureMatrix bgss(types::FeatureMatrix const& x) const;
    /** @brief Within-group sum of squares matrix (p × p). */
    types::FeatureMatrix wgss(types::FeatureMatrix const& x) const;

    /**
       * @brief Create a partition containing only the given groups.
       *
       * @param groups  Set of group labels to keep.
       * @return        New GroupPartition restricted to @p groups.
       */
    GroupPartition subset(GroupSet const& groups) const;

    using SplitSizes = std::map<types::Outcome, int>;

    /**
       * @brief Split each group's block into left and right children.
       *
       * For each leaf group, @p left_sizes specifies how many rows go to
       * the left child (the first rows of the block).  The remaining rows
       * go to the right child.  Groups absent from @p left_sizes go
       * entirely to the right child (left_count = 0).
       *
       * The caller is responsible for having already reordered rows within
       * each block so that left-bound observations come first.
       *
       * @param left_sizes  Maps each leaf group to its left child row count.
       * @return            Pair of {left, right} GroupPartitions.
       */
    std::pair<GroupPartition, GroupPartition> split(SplitSizes const& left_sizes) const;

    /**
       * @brief Merge groups according to a mapping.
       *
       * @param mapping  Maps original group labels to supergroup labels.
       * @return         New GroupPartition with merged groups.
       */
    GroupPartition remap(GroupMap const& mapping) const;

    /**
       * @brief Collapse all groups into a single supergroup.
       *
       * @return  New GroupPartition with one supergroup containing all groups.
       */
    GroupPartition collapse() const;

    /** @brief Set of all group labels in this partition. */
    GroupSet const groups;
    /** @brief Maps each group to its supergroup (identity if no merge). */
    GroupMap const supergroups;
    /** @brief Maps each group to its set of subgroups. */
    GroupInvMap const subgroups;

  private:
    struct Block {
      int start;
      int end;
      int size;
      std::optional<types::Outcome> next;
      std::optional<types::Outcome> prev;
    };

    using BlockMap = std::map<types::Outcome, Block>;
    BlockMap const Blocks;

    static BlockMap init_blocks(GroupVector const& y);
    static GroupMap init_supergroups(GroupSet const& groups);

    explicit GroupPartition(BlockMap const& Blocks);

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups);

    GroupPartition(BlockMap const& Blocks, GroupMap const& supergroups);

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups, GroupMap const& supergroups);
  };
}
