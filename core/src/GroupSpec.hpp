#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"

#include "Map.hpp"
#include "Invariant.hpp"

#include <optional>

namespace models::stats {
  template<typename T>
  using Data = math::DMatrix<T>;

  template<typename G>
  class GroupSpec {
    public:

      static bool is_contiguous(const DataColumn<G> &y) {
        std::set<G> visited;

        for (int i = 0; i < y.rows(); i++) {
          if (visited.count(y(i)) == 0) {
            visited.insert(y(i));
          } else if (y(i - 1) != y(i)) {
            return false;
          }
        }

        return true;
      }

      const std::set<G> groups;
      const std::map<G, G> supergroups;
      const std::map<G, std::set<G> > subgroups;

    private:

      struct Node {
        int start;
        int end;
        int size;
        std::optional<G> next;
        std::optional<G> prev;
      };

      const std::map<G, Node> nodes;

      std::map<G, Node> init_nodes(const DataColumn<G> &y) {
        std::map<G, Node> nodes;

        for (int i = 0; i < y.rows(); i++) {
          if (nodes.count(y(i)) == 0) {
            G curr = y(i);

            nodes[curr] = Node{ .start = i };

            if (i != 0) {
              G prev = y(i - 1);
              nodes[curr].prev = prev;
              nodes[prev].next = curr;
              nodes[prev].end  = i - 1;
              nodes[prev].size = i - nodes[prev].start;
            }
          } else if (y(i - 1) != y(i)) {
            throw std::invalid_argument("GroupSpec: data is not organized in contiguous groups");
          }
        }

        G last_group = y(y.rows() - 1);
        nodes.at(last_group).end  = y.rows() - 1;
        nodes.at(last_group).size = y.rows() - nodes.at(last_group).start;

        return nodes;
      }

      std::map<G, G> init_supergroups() {
        std::map<G, G> supergroups;

        for (const G &g : groups) {
          supergroups[g] = g;
        }

        return supergroups;
      }

      GroupSpec(
        const std::map<G, Node> &nodes,
        const std::set<G> &      groups) :
        groups(groups),
        nodes(nodes),
        supergroups(init_supergroups()),
        subgroups(utils::invert(supergroups)) {
      }

      GroupSpec(
        const std::map<G, Node> & nodes,
        const std::map<G, G> &    supergroups) :
        groups(utils::values(supergroups)),
        nodes(nodes),
        supergroups(supergroups),
        subgroups(utils::invert(supergroups)) {
      }

      GroupSpec(
        const std::map<G, Node> & nodes,
        const std::set<G> &       groups,
        const std::map<G, G> &    supergroups) :
        groups(groups),
        nodes(nodes),
        supergroups(supergroups),
        subgroups(utils::invert(supergroups)) {
      }

    public:

      GroupSpec(const DataColumn<G> &y) :
        groups(unique(y)),
        nodes(init_nodes(y)),
        supergroups(init_supergroups()),
        subgroups(utils::invert(supergroups)) {
      }

      int group_start(const G &group) const {
        return nodes.at(group).start;
      }

      int group_end(const G &group) const {
        return nodes.at(group).end;
      }

      int group_size(const G &group) const {
        return nodes.at(group).size;
      }

      template<typename T>
      auto group(const Data<T> &x, const G &group) const {
        std::vector<int> indices;

        std::set<G> subgroups = this->subgroups.at(group);

        for (const auto &g : subgroups) {
          for (int i = group_start(g); i <= group_end(g); i++) {
            invariant(i >= 0 && i < x.rows(), "GroupSpec::group: index out of bounds");
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::all);
      }

      template<typename T>
      auto data(const Data<T> &x) const {
        std::vector<int> indices;

        for (const auto &group : nodes) {
          for (int i = group_start(group.first); i <= group_end(group.first); i++) {
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::all);
      }

      template<typename T>
      DataColumn<T> mean(const Data<T> &x) const {
        return data(x).colwise().mean();
      }

      template<typename T>
      Data<T> bgss(const Data<T> &x) const {
        DataColumn<T> global_mean = this->mean(x);
        Data<T> result            = Data<T>::Zero(x.cols(), x.cols());

        std::set<G> groups = utils::values(this->supergroups);

        for (const G &g : groups) {
          auto group_data    = group(x, g);
          auto group_mean    = group_data.colwise().mean().transpose();
          auto centered_mean = group_mean - global_mean;

          result += group_data.rows() * (centered_mean * centered_mean.transpose());
        }

        return result;
      }

      template<typename T>
      Data<T> wgss(const Data<T> &x) const {
        Data<T> result = Data<T>::Zero(x.cols(), x.cols());

        std::set<G> groups = utils::values(this->supergroups);

        for (const G &g : groups) {
          auto group_data          = group(x, g);
          auto centered_group_data = group_data.rowwise() - group_data.colwise().mean();

          result += centered_group_data.transpose() * centered_group_data;
        }

        return result;
      }

      GroupSpec<G> subset(std::set<G> groups) const {
        std::map<G, Node> subset_nodes;

        G prev = -1;

        for (const auto &group : groups) {
          Node node = {
            .start  = nodes.at(group).start,
            .end    = nodes.at(group).end,
            .size   = nodes.at(group).size
          };

          if (prev != -1) {
            node.prev                  = prev;
            subset_nodes.at(prev).next = group;
          }

          subset_nodes[group] = node;

          prev = group;
        }

        return GroupSpec<G>(subset_nodes, groups);
      }

      GroupSpec<G> remap(const std::map<G, G> &mapping) const {
        return GroupSpec<G>(this->nodes, mapping);
      }
  };
}
