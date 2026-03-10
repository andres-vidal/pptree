/**
 * @file Timing.hpp
 * @brief High-resolution timing utility for measuring function execution time.
 */
#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace pptree::io {
  /**
   * @brief Return the current local time as an ISO 8601 string (YYYY-MM-DDTHH:MM:SS).
   */
  inline std::string now_iso8601() {
    auto now    = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm  = {};

    #ifdef _WIN32
    localtime_s(&tm, &time_t);
    #else
    localtime_r(&time_t, &tm);
    #endif

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
  }
  /**
   * @brief Measure the wall-clock execution time of a callable in milliseconds.
   *
   * @tparam F A callable (lambda, function pointer, std::function, etc.)
   * @param f The callable to time.
   * @return A pair of (result, duration_ms) where result is the return value
   *         of f() and duration_ms is the elapsed time in milliseconds.
   */
  template<typename F>
  auto measure_time_ms(F&& f) {
    auto str = std::chrono::high_resolution_clock::now();
    auto res = f();
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - str);

    return std::make_pair(std::move(res), dur.count());
  }
}
