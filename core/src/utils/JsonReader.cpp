#include "utils/JsonReader.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ppforest2 {
  namespace {
    std::string join_allowed(std::initializer_list<char const*> allowed) {
      std::string out;
      bool first = true;
      for (char const* a : allowed) {
        if (!first) out += ", ";
        first = false;
        out += a;
      }
      return out;
    }

    [[noreturn]] void throw_error(std::string const& path, std::string const& what) {
      throw std::runtime_error(path + ": " + what);
    }
  }

  JsonReader::JsonReader(nlohmann::json const& j, std::string path)
      : j_(j), path_(std::move(path)) {}

  std::string JsonReader::child_path(std::string const& key) const {
    // `path_` is the path to the current object; append `.key`. We don't
    // quote keys that contain dots — the model JSON doesn't use them.
    return path_.empty() ? key : path_ + "." + key;
  }

  void JsonReader::require_object() const {
    if (!j_.is_object()) {
      throw_error(path_, std::string("expected object, got ") + j_.type_name());
    }
  }

  nlohmann::json const& JsonReader::require_present(std::string const& key) const {
    require_object();
    auto it = j_.find(key);
    if (it == j_.end()) {
      throw_error(child_path(key), "missing required key");
    }
    return *it;
  }

  JsonReader JsonReader::at(std::string const& key) const {
    auto const& child = require_present(key);
    if (!child.is_object()) {
      throw_error(child_path(key), std::string("expected object, got ") + child.type_name());
    }
    return JsonReader(child, child_path(key));
  }

  bool JsonReader::contains(std::string const& key) const {
    return j_.is_object() && j_.contains(key);
  }

  template<typename T> T JsonReader::require(std::string const& key) const {
    auto const& v = require_present(key);
    try {
      return v.get<T>();
    } catch (nlohmann::json::type_error const&) {
      throw_error(child_path(key), std::string("unexpected type: ") + v.type_name());
    }
  }

  template<typename T> T JsonReader::optional(std::string const& key, T fallback) const {
    if (!contains(key)) return fallback;
    return require<T>(key);
  }

  std::string JsonReader::require_enum(
      std::string const& key, std::initializer_list<char const*> allowed
  ) const {
    auto const& v = require_present(key);
    if (!v.is_string()) {
      throw_error(child_path(key), std::string("expected string, got ") + v.type_name());
    }
    std::string const value = v.get<std::string>();
    for (char const* a : allowed) {
      if (value == a) return value;
    }
    throw_error(
        child_path(key),
        "must be one of [" + join_allowed(allowed) + "] (got '" + value + "')"
    );
  }

  long long JsonReader::require_int(std::string const& key, long long min, long long max) const {
    auto const& v = require_present(key);
    // Accept JSON integers outright; also accept numbers that happen to be
    // integer-valued (R serializes its `numeric` type as JSON number with no
    // integer marker, so `5L` from R round-trips as `5.0` on the wire).
    long long value;
    if (v.is_number_integer()) {
      value = v.get<long long>();
    } else if (v.is_number_float()) {
      double const d = v.get<double>();
      if (!std::isfinite(d) || d != std::floor(d)) {
        throw_error(child_path(key), "expected integer, got non-integer number (" + v.dump() + ")");
      }
      // `(double)LLONG_MAX` rounds up (LLONG_MAX is 2^63 - 1 but the closest
      // representable double is 2^63), so use strict-less comparisons against
      // the next representable powers-of-two to stay away from the UB edge
      // of `static_cast<long long>(d)` for out-of-range values.
      constexpr double kMin = -9.2233720368547758e18; // -2^63
      constexpr double kMax = 9.2233720368547758e18;  //  2^63
      if (d <= kMin || d >= kMax) {
        throw_error(
            child_path(key),
            "integer out of representable range (got " + v.dump() + ")"
        );
      }
      value = static_cast<long long>(d);
    } else {
      throw_error(child_path(key), std::string("expected integer, got ") + v.type_name());
    }
    if (value < min || value > max) {
      std::ostringstream oss;
      oss << "must be in [";
      if (min == std::numeric_limits<long long>::min()) oss << "-∞"; else oss << min;
      oss << ", ";
      if (max == std::numeric_limits<long long>::max()) oss << "∞"; else oss << max;
      oss << "] (got " << value << ")";
      throw_error(child_path(key), oss.str());
    }
    return value;
  }

  double JsonReader::require_number(std::string const& key, double min, double max) const {
    auto const& v = require_present(key);
    if (!v.is_number()) {
      throw_error(child_path(key), std::string("expected number, got ") + v.type_name());
    }
    double const value = v.get<double>();
    if (value < min || value > max) {
      std::ostringstream oss;
      oss << "must be in [";
      if (min == -std::numeric_limits<double>::infinity()) oss << "-∞"; else oss << min;
      oss << ", ";
      if (max == std::numeric_limits<double>::infinity()) oss << "∞"; else oss << max;
      oss << "] (got " << value << ")";
      throw_error(child_path(key), oss.str());
    }
    return value;
  }

  void JsonReader::only_keys(std::initializer_list<char const*> allowed) const {
    require_object();
    for (auto it = j_.begin(); it != j_.end(); ++it) {
      bool ok = false;
      for (char const* a : allowed) {
        if (it.key() == a) { ok = true; break; }
      }
      if (!ok) {
        throw_error(
            path_,
            "unexpected key '" + it.key() + "' (allowed: " + join_allowed(allowed) + ")"
        );
      }
    }
  }

  nlohmann::json const& JsonReader::require_array(std::string const& key) const {
    auto const& v = require_present(key);
    if (!v.is_array()) {
      throw_error(child_path(key), std::string("expected array, got ") + v.type_name());
    }
    return v;
  }

  // Explicit instantiations for the types we actually use.
  template int JsonReader::require<int>(std::string const&) const;
  template long long JsonReader::require<long long>(std::string const&) const;
  template float JsonReader::require<float>(std::string const&) const;
  template double JsonReader::require<double>(std::string const&) const;
  template std::string JsonReader::require<std::string>(std::string const&) const;
  template bool JsonReader::require<bool>(std::string const&) const;

  template int JsonReader::optional<int>(std::string const&, int) const;
  template long long JsonReader::optional<long long>(std::string const&, long long) const;
  template float JsonReader::optional<float>(std::string const&, float) const;
  template double JsonReader::optional<double>(std::string const&, double) const;
  template std::string JsonReader::optional<std::string>(std::string const&, std::string) const;
  template bool JsonReader::optional<bool>(std::string const&, bool) const;
}
