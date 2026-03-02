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

namespace pptree::stats {
  class GroupPartition {
    using Group       = types::Response;
    using GroupSet    = std::set<types::Response>;
    using GroupMap    = std::map<types::Response, types::Response>;
    using GroupInvMap = std::map<types::Response, GroupSet>;
    using GroupVector = types::Vector<types::Response>;

    public:
      static bool is_contiguous(const GroupVector& y);

      GroupPartition(const types::ResponseVector& y);

      int group_start(const Group& group) const;
      int group_end(const Group& group) const;
      int group_size(const Group& group) const;


      auto group(const types::FeatureMatrix& x, const Group& group) const {
        std::vector<int> indices;

        const auto& subs = this->subgroups.at(group);

        for (const auto& g : subs) {
          for (int i = group_start(g); i <= group_end(g); ++i) {
            invariant(i >= 0 && i < x.rows(),
              "GroupPartition::group: index out of bounds");
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::all);
      }

      auto data(const types::FeatureMatrix& x) const {
        std::vector<int> indices;

        for (const auto& kv : Blocks) {
          const auto& g = kv.first;
          for (int i = group_start(g); i <= group_end(g); ++i) {
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::all);
      }

      types::FeatureVector mean(const types::FeatureMatrix& x) const;
      types::FeatureMatrix bgss(const types::FeatureMatrix& x) const;
      types::FeatureMatrix wgss(const types::FeatureMatrix& x) const;

      GroupPartition subset(GroupSet groups) const;
      GroupPartition remap(const GroupMap& mapping) const;

      const GroupSet groups;
      const GroupMap supergroups;
      const GroupInvMap subgroups;

    private:
      struct Block {
        int start;
        int end;
        int size;
        std::optional<types::Response> next;
        std::optional<types::Response> prev;
      };

      using BlockMap = std::map<types::Response, Block>;
      const BlockMap Blocks;

      BlockMap init_Blocks(const GroupVector& y);
      GroupMap init_supergroups();

      GroupPartition(
        const BlockMap& Blocks,
        const GroupSet& groups);

      GroupPartition(
        const BlockMap& Blocks,
        const GroupMap& supergroups);

      GroupPartition(
        const BlockMap& Blocks,
        const GroupSet& groups,
        const GroupMap& supergroups);
  };
}
