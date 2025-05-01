#pragma once

#include "TreeNodeVisitor.hpp"

#include "Data.hpp"
#include "DataColumn.hpp"
#include "SortedDataSpec.hpp"
#include "Projector.hpp"
#include "TrainingSpec.hpp"

#include <nlohmann/json.hpp>

namespace models {
  using json = nlohmann::json;
  template<typename T>
  using Threshold = T;
  template<typename T, typename R>
  struct TreeNode {
    virtual ~TreeNode()                                       = default;
    virtual void accept(TreeNodeVisitor<T, R> &visitor) const = 0;
    virtual R predict(const stats::DataColumn<T> &data) const = 0;
    virtual R response() const                                = 0;
    virtual json to_json() const                              = 0;

    virtual bool equals(const TreeNode<T, R> &other) const = 0;

    bool operator==(const TreeNode<T, R> &other) const {
      return this->equals(other);
    }

    bool operator!=(const TreeNode<T, R> &other) const {
      return !this->equals(other);
    }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const TreeNode<T, R> &node) {
    return ostream << node.to_json().dump(2, ' ', false);
  }
}
