#pragma once

#include "TreeNode.hpp"

namespace models {
  template<typename T, typename R>
  struct Response : public Node<T, R> {
    R value;

    explicit Response(R value) : value(value) {
    }

    void accept(NodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      return value;
    }

    R predict(const stats::DataColumn<T> &data) const override {
      return value;
    }

    bool operator==(const Response<T, R> &other) const {
      return value == other.value;
    }

    bool operator!=(const Response<T, R> &other) const {
      return !(*this == other);
    }

    json to_json() const override {
      return json{
        { "value", value }
      };
    }

    bool equals(const Node<T, R> &other) const override {
      return other.equals(*this);
    }

    bool equals(const Condition<T, R> &other) const override {
      return false;
    }

    bool equals(const Response<T, R> &other) const override {
      return *this == other;
    }
  };
}
