#pragma once

#include <iostream>
#include <map>

#include <nlohmann/json.hpp>

using json = nlohmann::json;


#ifdef NDEBUG
  #define LOG_INFO    if (false) std::cout
  #define LOG_DEBUG   if (false) std::cout
  #define LOG_WARNING if (false) std::cout
#else
  #define LOG_INFO    std::cout << "[INFO]" << "[" << __FUNCTION__ << "] "
  #define LOG_DEBUG   std::cout << "[DEBUG]" << "[" << __FUNCTION__ << "] "
  #define LOG_WARNING std::cout << "[WARNING]" << "[" << __FUNCTION__ << "] "
#endif

namespace models {
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
}
