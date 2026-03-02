/**
 * @file Color.hpp
 * @brief TTY-aware colored terminal output utilities.
 *
 * Wraps the fmt/color.h library with conditional formatting that
 * automatically disables ANSI escape codes when stdout is not a TTY
 * or when the user passes the --no-color flag. On Windows, enables
 * Virtual Terminal Processing for ANSI support.
 */
#pragma once

#include <string>

#include <fmt/color.h>
#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace pptree::cli {
  /**
   * @brief Global toggle for colored output.
   * @return Reference to the static enabled flag.
   */
  inline bool& color_enabled() {
    static bool enabled = true;
    return enabled;
  }

  /**
   * @brief Initialize color support based on TTY detection and user preference.
   *
   * On Windows, enables ANSI Virtual Terminal Processing. Disables color
   * when stdout is not a TTY or when the user explicitly requests no color.
   *
   * @param no_color If true, unconditionally disable colored output.
   */
  inline void init_color(bool no_color) {
    if (no_color) {
      color_enabled() = false;
      return;
    }

    #ifdef _WIN32
    HANDLE hOut  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;

    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
      SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      color_enabled() = true;
    } else {
      color_enabled() = false;
    }

    #else
    color_enabled() = isatty(fileno(stdout));
    #endif
  }

  /**
   * @brief Format text in red (for error messages).
   * @param s The string to colorize.
   * @return The colored string, or the original string if color is disabled.
   */
  inline std::string error(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fg(fmt::terminal_color::red), "{}", s);
  }

  /**
   * @brief Format text in green (for success messages).
   * @param s The string to colorize.
   * @return The colored string, or the original string if color is disabled.
   */
  inline std::string success(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fg(fmt::terminal_color::green), "{}", s);
  }

  /**
   * @brief Format text in bold (for emphasis / labels).
   * @param s The string to emphasize.
   * @return The bold string, or the original string if color is disabled.
   */
  inline std::string emphasis(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fmt::emphasis::bold, "{}", s);
  }

  /**
   * @brief Format text in dim gray (for hints and secondary info).
   * @param s The string to dim.
   * @return The dimmed string, or the original string if color is disabled.
   */
  inline std::string muted(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fg(fmt::terminal_color::bright_black), "{}", s);
  }

  /**
   * @brief Format text in cyan (for informational highlights like progress bars).
   * @param s The string to colorize.
   * @return The colored string, or the original string if color is disabled.
   */
  inline std::string info(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fg(fmt::terminal_color::cyan), "{}", s);
  }

  /**
   * @brief Format text in yellow (for warnings).
   * @param s The string to colorize.
   * @return The colored string, or the original string if color is disabled.
   */
  inline std::string warning(const std::string& s) {
    if (!color_enabled()) return s;

    return fmt::format(fg(fmt::terminal_color::yellow), "{}", s);
  }
}
