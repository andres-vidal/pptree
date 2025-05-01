#pragma once

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
  struct Node;
  template<typename T, typename R>
  struct Condition;
  template<typename T, typename R>
  struct Response;
  template<typename T, typename R>
  struct NodeVisitor {
    virtual void visit(const Condition<T, R> &condition) = 0;
    virtual void visit(const Response<T, R> &response)   = 0;
  };

  template<typename T, typename R>
  struct Node {
    virtual ~Node()                                           = default;
    virtual void accept(NodeVisitor<T, R> &visitor) const     = 0;
    virtual R predict(const stats::DataColumn<T> &data) const = 0;
    virtual R response() const                                = 0;
    virtual json to_json() const                              = 0;
    virtual bool equals(const Node<T, R> &other) const        = 0;
    virtual bool equals(const Condition<T, R> &other) const   = 0;
    virtual bool equals(const Response<T, R> &other) const    = 0;


    bool operator==(const Node<T, R> &other) const {
      return this->equals(other);
    }

    bool operator!=(const Node<T, R> &other) const {
      return !this->equals(other);
    }
  };



  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Node<T, R> &node) {
    return ostream << node.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
    return ostream << condition.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
    return ostream << response.to_json().dump(2, ' ', false);
  }
}
