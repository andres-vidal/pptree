#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"

#include <optional>

namespace models::stats {
  template<typename T>
  using Data = math::DMatrix<T>;

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
      const std::set<G> groups;


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
        const std::map<G, Node> &nodes,
        const std::set<G> &      groups) :
        x(x),
        groups(groups),
        nodes(nodes) {
      }

    public:

      GroupSpec(const Data<T> &x, const DataColumn<G> &y) :
        x(x),
        groups(unique(y)),
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
          DataView<T> group_data      = group(g);
          DataColumn<T> group_mean    = group_data.colwise().mean();
          DataColumn<T> centered_mean = group_mean - global_mean;

          result.noalias() += group_data.rows() * (centered_mean * centered_mean.transpose());
        }

        return result;
      }

      Data<T> wgss() const {
        Data<T> result = Data<T>::Zero(this->cols(), this->cols());

        for (const G &g : this->groups) {
          DataView<T> group_data      = group(g);
          Data<T> centered_group_data = group_data.rowwise() - group_data.colwise().mean();

          result.noalias() += centered_group_data.transpose() * centered_group_data;
        }

        return result;
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

        return GroupSpec<T, G>(this->x, subset_nodes, groups);
      }

      GroupSpec<T, G> analog(const Data<T>& data) const {
        return GroupSpec<T, G>(data, this->nodes, this->groups);
      }

      std::set<G> classes() const {
        return this->groups;
      }
  };
}
