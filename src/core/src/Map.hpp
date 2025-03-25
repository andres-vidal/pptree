#pragma once

#include <vector>
#include <map>

namespace utils {
  template<typename K, typename V>
  std::vector<K> sort_keys_by_value(const std::map<K, V> &mapping) {
    std::vector<std::pair<K, V> > items(mapping.begin(), mapping.end());

    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
      });

    std::vector<K> result;
    result.reserve(items.size());
    for (auto &kv : items) {
      result.push_back(kv.first);
    }

    return result;
  }

  template<typename K, typename V>
  std::map<V, std::set<K> > invert(const std::map<K, V> &map) {
    std::map<V, std::set<K> > result;

    for (const auto&[key, value] : map) {
      result[value].insert(key);
    }

    return result;
  }
}
