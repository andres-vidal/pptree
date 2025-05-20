#pragma once

#include <vector>
#include <map>
#include <set>

namespace utils {
  template<typename K, typename V>
  std::map<V, std::set<K> > invert(const std::map<K, V> &map) {
    std::map<V, std::set<K> > result;

    for (const auto&[key, value] : map) {
      result[value].insert(key);
    }

    return result;
  }

  template<typename K, typename V>
  std::set<K> keys(const std::map<K, V> &map) {
    std::set<K> result;

    for (const auto&[key, value] : map) {
      result.insert(key);
    }

    return result;
  }

  template<typename K, typename V>
  std::set<V> values(const std::map<K, V> &map) {
    std::set<V> result;

    for (const auto&[key, value] : map) {
      result.insert(value);
    }

    return result;
  }
}
