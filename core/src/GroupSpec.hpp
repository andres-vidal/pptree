#pragma once

#include "DMatrix.hpp"

#include <optional>

namespace models::stats {
  template<typename T>
  using Data = math::DMatrix<T>;

  template<typename T>
  using DataColumn = math::DVector<T>;

  template <typename T>
  using DataView = Eigen::Block<const Data<T> >;

  template<typename T, typename G>
  class GroupSpec {
    private:

      struct Node {
        int start;
        int end;
        std::optional<G> next;
        std::optional<G> prev;
      };

      const Data<T> x;
      const std::map<G, Node> nodes;

      std::map<G, Node> init_nodes(const DataColumn<G> &y) {
        std::map<G, Node> nodes;

        G curr = -1;

        for (int i = 0; i < y.rows(); i++) {
          if (nodes.count(y(i)) == 0) {
            curr = y(i);

            nodes[curr] = Node{ .start = i };

            if (i != 0) {
              G prev = y(i - 1);
              nodes[curr].prev = prev;
              nodes[prev].next = curr;
              nodes[prev].end  = i - 1;
            }
          } else if (curr != y(i)) {
            throw std::invalid_argument("GroupSpec: data is not organized in contiguous groups");
          }
        }

        nodes.at(curr).end = this->x.rows() - 1;

        return nodes;
      }

      GroupSpec(
        const Data<T> &          x,
        const std::map<G, Node> &nodes
        ) :
        x(x),
        nodes(nodes) {
      }

    public:

      GroupSpec(const Data<T> &x, const DataColumn<G> &y) :
        x(x),
        nodes(init_nodes(y)) {
      }

      int group_start(const G &group) const {
        return nodes.at(group).start;
      }

      int group_end(const G &group) const {
        return nodes.at(group).end;
      }

      int group_size(const G &group) const {
        return 1 + group_end(group) - group_start(group);
      }

      DataView<T> group(const G &group) const {
        return this->x(Eigen::seq(group_start(group), group_end(group)), Eigen::all);
      }

      GroupSpec<T, G> subset(std::set<G> groups) const {
        std::map<G, Node> subset_nodes;

        G prev = -1;

        for (const auto &group : groups) {
          Node node = {
            .start  = nodes.at(group).start,
            .end    = nodes.at(group).end,
          };

          if (prev != -1) {
            node.prev                  = prev;
            subset_nodes.at(prev).next = group;
          }

          subset_nodes[group] = node;

          prev = group;
        }

        return GroupSpec<T, G>(this->x, subset_nodes);
      }
  };
}
