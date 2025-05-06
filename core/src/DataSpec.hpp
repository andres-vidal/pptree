#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"

#include "Map.hpp"

#include <optional>

namespace models::stats {
  template<typename T>
  using Data = math::DMatrix<T>;

  template<typename T, typename G>
  class DataSpec {
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

      const Data<T> &x;
      const std::set<G> groups;

    private:

      struct Node {
        int start;
        int end;
        int size;
        std::optional<G> next;
        std::optional<G> prev;
      };

      const std::map<G, Node> nodes;
      const std::map<G, G> supergroups;
      const std::map<G, std::set<G> > subgroups;

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
            throw std::invalid_argument("DataSpec: data is not organized in contiguous groups");
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

      DataSpec(
        const Data<T> &          x,
        const std::map<G, Node> &nodes,
        const std::set<G> &      groups) :
        x(x),
        groups(groups),
        nodes(nodes),
        supergroups(init_supergroups()),
        subgroups(utils::invert(supergroups)) {
      }

      DataSpec(
        const Data<T> &          x,
        const std::map<G, Node> &nodes,
        const std::map<G, G> &   supergroups) :
        x(x),
        groups(utils::values(supergroups)),
        nodes(nodes),
        supergroups(supergroups),
        subgroups(utils::invert(supergroups)) {
      }

      DataSpec(
        const Data<T> &          x,
        const std::map<G, Node> &nodes,
        const std::set<G> &      groups,
        const std::map<G, G> &   supergroups) :
        x(x),
        groups(groups),
        nodes(nodes),
        supergroups(supergroups),
        subgroups(utils::invert(supergroups)) {
      }

    public:

      DataSpec(const Data<T> &x, const DataColumn<G> &y) :
        x(x),
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

      auto group(const G &group) const {
        std::vector<int> indices;

        std::set<G> subgroups = this->subgroups.at(group);

        for (const auto &g : subgroups) {
          for (int i = group_start(g); i <= group_end(g); i++) {
            indices.push_back(i);
          }
        }

        return this->x(indices, Eigen::all);
      }

      auto data() const {
        std::vector<int> indices;

        for (const auto &group : nodes) {
          for (int i = group_start(group.first); i <= group_end(group.first); i++) {
            indices.push_back(i);
          }
        }

        return this->x(indices, Eigen::all);
      }

      int rows() const {
        return data().rows();
      }

      int cols() const {
        return this->x.cols();
      }

      DataColumn<T> mean() const {
        return data().colwise().mean();
      }

      Data<T> bgss() const {
        DataColumn<T> global_mean = this->mean();
        Data<T> result            = Data<T>::Zero(this->cols(), this->cols());

        for (const G &g : this->groups) {
          auto group_data             = group(g);
          DataColumn<T> group_mean    = group_data.colwise().mean();
          DataColumn<T> centered_mean = group_mean - global_mean;

          result.noalias() += group_data.rows() * (centered_mean * centered_mean.transpose());
        }

        return result;
      }

      Data<T> wgss() const {
        Data<T> result = Data<T>::Zero(this->cols(), this->cols());

        for (const G &g : this->groups) {
          auto group_data          = group(g);
          auto centered_group_data = group_data.rowwise() - group_data.colwise().mean();

          result.noalias() += centered_group_data.transpose() * centered_group_data;
        }

        return result;
      }

      DataSpec<T, G> subset(std::set<G> groups) const {
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

        return DataSpec<T, G>(this->x, subset_nodes, groups);
      }

      DataSpec<T, G> analog(const Data<T>& data) const {
        return DataSpec<T, G>(data, this->nodes, this->groups, this->supergroups);
      }

      DataSpec<T, G> remap(const std::map<G, G> &mapping) const {
        return DataSpec<T, G>(this->x, this->nodes, mapping);
      }
  };
}
