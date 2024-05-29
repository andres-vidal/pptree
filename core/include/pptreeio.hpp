#pragma once

#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

#ifdef NDEBUG
  #define LOG_INFO    if (false) std::cout
  #define LOG_DEBUG   if (false) std::cout
  #define LOG_WARNING if (false) std::cout
#else
  #define LOG_INFO    std::cout << "[INFO]" << "[" << __FUNCTION__ << "] "
  #define LOG_DEBUG   std::cout << "[DEBUG]" << "[" << __FUNCTION__ << "] "
  #define LOG_WARNING std::cout << "[WARNING]" << "[" << __FUNCTION__ << "] "
#endif

using json = nlohmann::json;

template<typename T, typename R, typename D >
struct Tree;
template <typename T, typename R>
struct Node;
template<typename T, typename R >
struct Response;
template<typename T, typename R >
struct Condition;
template<typename T, typename R >
struct Forest;

template<typename T, typename R >
void to_json(json& j, const Condition<T, R> &condition);
template<typename T, typename R >
void to_json(json& j, const Response<T, R> &response);
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
    to_json(j, node.as_response());
  } else {
    to_json(j, node.as_condition());
  }
}

template<typename T, typename R, typename D>
void to_json(json& j, const Tree<T, R, D>& tree) {
  j = json{
    { "root", *tree.root }
  };
}

template<typename T, typename R>
void to_json(json& j, const Forest<T, R>& forest) {
  std::vector<json> trees_json;

  for (const auto& tree : forest.trees) {
    json tree_json;
    to_json(tree_json, *tree);
    trees_json.push_back(tree_json);
  }

  j = json{ { "trees", trees_json } };
}

template<typename T, typename R, typename D>
std::ostream& operator<<(std::ostream & ostream, const Tree<T, R, D>& tree) {
  json json_tree(tree);
  return ostream << json_tree.dump(2, ' ', false);
}

template<typename T, typename R>
std::ostream& operator<<(std::ostream & ostream, const Node<T, R> &node) {
  json json_node(node);
  return ostream << json_node.dump(2, ' ', false);
}

template<typename T, typename R>
std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
  json json_condition(condition);
  return ostream << json_condition.dump(2, ' ', false);
}

template<typename T, typename R>
std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
  json json_response(response);
  return ostream << json_response.dump(2, ' ', false);
}

template<typename T, typename R>
std::ostream& operator<<(std::ostream & ostream, const Forest<T, R>& forest) {
  json json_response(forest);
  return ostream << json_response.dump(2, ' ', false);
}

template<typename V>
std::ostream& operator<<(std::ostream& ostream, const std::vector<V> &vec) {
  json json_vector(vec);
  return ostream << json_vector.dump();
}

template<typename V, typename C1, typename C2>
std::ostream& operator<<(std::ostream& ostream, const std::set<V, C1, C2> &set) {
  json json_set(set);
  return ostream << json_set.dump();
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& ostream, const std::map<K, V> &map) {
  json json_map(map);
  return ostream << json_map.dump();
}

template<typename V>
std::ostream& operator<<(std::ostream& ostream, const std::map<int, V> &map) {
  std::map<std::string, V> string_map;

  for (const auto& [key, val] : map) {
    string_map[std::to_string(key)] = val;
  }

  json json_map(string_map);
  return ostream << json_map.dump();
}
