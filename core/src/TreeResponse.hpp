#pragma once

#include "TreeNode.hpp"

namespace models {
  template<typename T, typename R>
  struct TreeResponse : public TreeNode<T, R> {
    R value;

    explicit TreeResponse(R value) : value(value) {
    }

    void accept(TreeNodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      return value;
    }

    R predict(const stats::DataColumn<T> &data) const override {
      return value;
    }

    bool equals(const TreeNode<T, R> &other) const override {
      const auto *resp = dynamic_cast<const TreeResponse<T, R> *>(&other);
      return resp && (value == resp->value);
    }

    json to_json() const override {
      return json{
        { "value", value }
      };
    }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const TreeResponse<T, R>& response) {
    return ostream << response.to_json().dump(2, ' ', false);
  }
}
