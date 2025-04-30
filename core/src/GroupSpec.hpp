#pragma once

#include "Data.hpp"

#include <optional>

namespace models::stats {
  template<typename T, typename G>
  class GroupSpec {
    private:

      struct Node {
        int index;
        std::optional<G> next;
        std::optional<G> prev;
      };

      const Data<T> x;
      const std::map<G, Node> nodes;

      std::map<G, Node> init_nodes(const DataColumn<G> &y) {
        std::map<G, Node> nodes;

        for (int i = 0; i < y.rows(); i++) {
          if (nodes.count(y(i)) == 0) {
            G curr = y(i);

            nodes[curr] = Node{ i };

            if (i != 0) {
              G prev = y(i - 1);
              nodes[curr].prev = prev;
              nodes[prev].next = curr;
            }
          }
        }

        return nodes;
      }

    public:

      GroupSpec(const Data<T> &x, const DataColumn<G> &y) :
        x(x),
        nodes(init_nodes(y)) {
      }

      int group_start(const G &group) const {
        return nodes.at(group).index;
      }

      int group_end(const G &group) const {
        std::optional<G> next = nodes.at(group).next;

        if (!next.has_value()) {
          return this->x.rows() - 1;
        }

        return nodes.at(next.value()).index - 1;
      }

      int group_size(const G &group) const {
        return 1 + group_end(group) - group_start(group);
      }

      DataView<T> group(const G &group) const {
        return this->x(Eigen::seq(group_start(group), group_end(group)), Eigen::all);
      }
  };
}
