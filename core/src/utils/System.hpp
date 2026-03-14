/**
 * @file System.hpp
 * @brief System-level utilities (process memory measurement).
 */
#pragma once

/** @brief System-level utilities (process memory measurement). */
namespace ppforest2::sys {
  /**
   * @brief Get the peak resident set size (RSS) of the current process in bytes.
   *
   * Uses platform-specific APIs: `GetProcessMemoryInfo` on Windows,
   * `getrusage` on POSIX (macOS reports bytes, Linux reports KB * 1024).
   *
   * @return Peak RSS in bytes, or -1 on failure.
   */
  long get_peak_rss_bytes();
}
