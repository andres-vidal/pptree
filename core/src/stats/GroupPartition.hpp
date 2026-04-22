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
   *   GroupPartition y_part(y);
   *
   *   // Extract rows belonging to group 0:
   *   auto x_group0 = y_part.group(x, 0);
   *
   *   // Between- and within-group statistics:
   *   auto B = y_part.bgss(x);   // between-group sum of squares (p × p)
   *   auto W = y_part.wgss(x);   // within-group sum of squares  (p × p)
   *
   *   // Restrict to a subset of groups:
   *   GroupPartition sub = y_part.subset({0, 2});
   * @endcode
   */
  class GroupPartition {
    using Group       = types::GroupId;
    using GroupSet    = std::set<types::GroupId>;
    using GroupMap    = std::map<types::GroupId, types::GroupId>;
    using GroupInvMap = std::map<types::GroupId, GroupSet>;
    using GroupVector = types::GroupIdVector;

  public:
    /** @brief Check whether all equal values in @p y form a single contiguous block. */
    static bool is_contiguous(GroupVector const& y);

    /** @brief Check contiguity of an `OutcomeVector` (float-encoded labels). */
    static bool is_contiguous(types::OutcomeVector const& y);

    /**
       * @brief Construct from a sorted response vector.
       *
       * @param y  Outcome vector (n) with contiguous group blocks.
       */
    GroupPartition(types::GroupIdVector const& y);

    /**
     * @brief Construct from a float-typed response vector.
     *
     * Classification `y` is carried as `OutcomeVector` (float) throughout the
     * training pipeline; this overload casts to integer labels internally
     * before building the block map. Values must encode integer labels.
     */
    GroupPartition(types::OutcomeVector const& y);

    /**
     * @brief Create a 2-group partition at specific row ranges.
     *
     * Both groups must be adjacent: start1 == end0 + 1.
     * Used by ByCutpoint to create regression child partitions
     * at arbitrary positions in the shared data matrix.
     *
     * @param start0  First row of group 0 (inclusive).
     * @param end0    Last row of group 0 (inclusive).
     * @param start1  First row of group 1 (inclusive).
     * @param end1    Last row of group 1 (inclusive).
     * @return        GroupPartition with groups {0, 1}.
     */
    static GroupPartition two_groups(int start0, int end0, int start1, int end1);

    /**
     * @brief Create a single-group partition at a specific row range.
     *
     * Used when a regression child has too few observations for a
     * meaningful 2-group split. The stop rule will turn this into a leaf.
     *
     * @param start  First row (inclusive).
     * @param end    Last row (inclusive).
     * @return       GroupPartition with group {0}.
     */
    static GroupPartition single_group(int start, int end);

    /** @brief First row index of the block for @p group. */
    int group_start(Group const& group) const;
    /** @brief Last row index (inclusive) of the block for @p group. */
    int group_end(Group const& group) const;
    /** @brief Number of observations in @p group. */
    int group_size(Group const& group) const;

    /**
     * @brief Total number of observations across all groups in the partition.
     *
     * Used by the tree builder to detect "no-progress" grouping splits:
     * if a child partition covers the same row count as its parent, the
     * split failed to partition the data and the builder converts the node
     * to a leaf to avoid unbounded recursion.
     */
    int total_size() const;

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

    using SplitSizes = std::map<types::GroupId, int>;

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
      std::optional<types::GroupId> next;
      std::optional<types::GroupId> prev;
    };

    using BlockMap = std::map<types::GroupId, Block>;
    BlockMap const Blocks;

    static BlockMap init_blocks(GroupVector const& y);
    static GroupMap init_supergroups(GroupSet const& groups);

    explicit GroupPartition(BlockMap const& Blocks);

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups);

    GroupPartition(BlockMap const& Blocks, GroupMap const& supergroups);

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups, GroupMap const& supergroups);
  };
}
