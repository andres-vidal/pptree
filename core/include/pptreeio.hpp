#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace pptree {
  template<typename T, typename R >
  struct Tree;
  template<typename T, typename R >
  struct Node;
  template<typename T, typename R >
  struct Response;

  template<typename T, typename R >
  void to_json(json& j, const Condition<T, R>& condition);
  template<typename T, typename R >
  void to_json(json& j, const Response<T, R>& response);
  template<typename T, typename R >
  void to_json(json& j, const Node<T, R> &node);

  template<typename T, typename R >
  void to_json(json& j, const Condition<T, R>& condition) {
    j = json{
      { "projector", condition.projector },
      { "threshold", condition.threshold },
      { "lower", *condition.lower },
      { "upper", *condition.upper }
    };
  }

  template<typename T, typename R >
  void to_json(json& j, const Response<T, R>& response) {
    j = json{
      { "value", response.value }
    };
  }

  template<typename T, typename R >
  void to_json(json& j, const Node<T, R>& node) {
    if (node.is_response()) {
      to_json(j, as_response<T, R>(node));
    } else {
      to_json(j, as_condition<T, R>(node));
    }
  }

  template<typename T, typename R>
  void to_json(json& j, const Tree<T, R>& tree) {
    j = json{
      { "root", *tree.root }
    };
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Tree<T, R>& tree) {
    json json_tree(tree);
    return ostream << json_tree;
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Node<T, R> &node) {
    json json_node(node);
    return ostream << json_node;
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
    json json_condition(condition);
    return ostream << json_condition;
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
    json json_response(response);
    return ostream << json_response;
  }

  template<typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::vector<V> &vec) {
    json json_vector(vec);
    return ostream << json_vector.dump();
  }

  template<typename K, typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::map<K, V> &map) {
    json json_map(map);
    return ostream << json_map.dump();
  }

  template<typename V>
  std::ostream& operator<<(std::ostream& ostream, const std::map<int, V> &map) {
    std::map<std::string, V> string_map;

    for (auto const& [key, val] : map) {
      string_map[std::to_string(key)] = val;
    }

    json json_map(string_map);
    return ostream << json_map.dump();
  }
}
