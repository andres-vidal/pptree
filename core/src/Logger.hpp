#pragma once

#include <iostream>
#include <fstream>

#include <map>
#include <set>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace models {
  using json = nlohmann::json;

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

  inline void log(std::stringstream &ss) {
    #ifdef _OPENMP
    std::string filename = "log/t" + std::to_string(omp_get_thread_num()) + ".txt";
    std::ofstream debug_file(filename, std::ios::app);

    if (debug_file.is_open()) {
      debug_file << ss.str();
      debug_file.close();
    }

    #else
    std::cout << ss.str();
    #endif
  }
}
