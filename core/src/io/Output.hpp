/**
 * @file Output.hpp
 * @brief Quiet-aware output helpers for CLI subcommands.
 *
 * Thin wrapper around fmt::print that respects the --quiet flag,
 * plus convenience methods for common CLI output patterns (file
 * save confirmations, error-catch wrappers).
 */
#pragma once

#include "io/Color.hpp"

#include <fmt/format.h>
#include <functional>
#include <string>

namespace pptree::io {
  /**
   * @brief Quiet-aware output context for CLI subcommands.
   *
   * Wraps a single boolean and provides methods that suppress output
   * when quiet mode is active. Construct once per subcommand from
   * CLIOptions::quiet and use throughout.
   */
  struct Output {
    bool quiet;

    explicit Output(bool quiet) : quiet(quiet) {
    }

    /**
     * @brief Print a formatted message, suppressed in quiet mode.
     */
    template<typename ... Args>
    void print(fmt::format_string<Args...> fmt_str, Args&&... args) const {
      if (!quiet) {
        fmt::print(fmt_str, std::forward<Args>(args)...);
      }
    }

    /**
     * @brief Print a file-save confirmation in the standard format.
     *
     * Produces: "{label} saved to {path}" with green prefix.
     * Suppressed in quiet mode.
     */
    void saved(const std::string& label, const std::string& path) const {
      if (!quiet) {
        fmt::print("{} saved to {}\n", label, path);
      }
    }
  };

  /**
   * @brief Run a callable, catch std::exception, print error, return 1.
   *
   * If the callable succeeds, returns 0. If it throws, prints the
   * error message to stderr with red "Error:" prefix and returns 1.
   *
   * @param f       Callable that may throw.
   * @param context Optional prefix for the error message.
   * @return 0 on success, 1 on caught exception.
   */
  inline int try_or_fail(
    const std::function<void()>& f,
    const std::string&           context = "") {
    try {
      f();
      return 0;
    } catch (const std::exception& e) {
      if (context.empty()) {
        fmt::print(stderr, "{} {}\n", error("Error:"), e.what());
      } else {
        fmt::print(stderr, "{} {}: {}\n", error("Error:"), context, e.what());
      }

      return 1;
    }
  }
}
