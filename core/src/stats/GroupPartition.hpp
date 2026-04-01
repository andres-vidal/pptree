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
    using Group       = types::Response;
    using GroupSet    = std::set<types::Response>;
    using GroupMap    = std::map<types::Response, types::Response>;
    using GroupInvMap = std::map<types::Response, GroupSet>;
    using GroupVector = types::Vector<types::Response>;

  public:
    /** @brief Check whether all equal values in @p y form a single contiguous block. */
    static bool is_contiguous(GroupVector const& y);

    /**
       * @brief Construct from a sorted response vector.
       *
       * @param y  Response vector (n) with contiguous group blocks.
       */
    GroupPartition(types::ResponseVector const& y);

    /** @brief First row index of the block for @p group. */
    int group_start(Group const& group) const;
    /** @brief Last row index (inclusive) of the block for @p group. */
    int group_end(Group const& group) const;
    /** @brief Number of observations in @p group. */
    int group_size(Group const& group) const;

    /**
       * @brief Extract rows belonging to a group (or supergroup).
       *
       * @param x      Feature matrix (n × p).
       * @param group  Group label.
       * @return       Sub-matrix of rows belonging to @p group.
       */
    auto group(types::FeatureMatrix const& x, Group const& group) const {
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
    auto data(types::FeatureMatrix const& x) const {
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
    GroupPartition subset(GroupSet groups) const;

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
      std::optional<types::Response> next;
      std::optional<types::Response> prev;
    };

    using BlockMap = std::map<types::Response, Block>;
    BlockMap const Blocks;

    BlockMap init_Blocks(GroupVector const& y);
    GroupMap init_supergroups();

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups);

    GroupPartition(BlockMap const& Blocks, GroupMap const& supergroups);

    GroupPartition(BlockMap const& Blocks, GroupSet const& groups, GroupMap const& supergroups);
  };
}
