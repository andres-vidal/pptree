/**
 * @file Table.hpp
 * @brief ANSI-safe column-driven table formatting.
 *
 * Provides primitives for building aligned tables that correctly
 * handle strings containing ANSI escape codes. Column widths are
 * defined once and applied consistently across header, separator,
 * and data rows.
 */
#pragma once

#include <string>
#include <vector>

namespace pptree::io {
  enum class Align { left, right };

  /**
   * @brief Compute the visual width of a string, ignoring ANSI escape codes.
   *
   * Skips any ESC [ ... m (SGR) sequences when counting characters.
   */
  inline int visual_width(const std::string& s) {
    int width     = 0;
    bool in_escape = false;

    for (char c : s) {
      if (in_escape) {
        if (c == 'm') in_escape = false;
      } else if (c == '\033') {
        in_escape = true;
      } else {
        ++width;
      }
    }

    return width;
  }

  /**
   * @brief Pad a string to a target visual width, handling ANSI codes.
   *
   * Unlike fmt::format width specifiers, this correctly handles strings
   * containing ANSI escape codes by computing visual width separately.
   */
  inline std::string pad(const std::string& s, int width, Align align = Align::right) {
    int gap = width - visual_width(s);

    if (gap <= 0) return s;

    std::string spaces(static_cast<std::size_t>(gap), ' ');

    return align == Align::right ? spaces + s : s + spaces;
  }

  /**
   * @brief Column definition for table formatting.
   */
  struct Column {
    std::string label;
    int         width;
    Align       align = Align::right;
  };

  /** @brief A row of pre-formatted cell strings. */
  using Row = std::vector<std::string>;

  /**
   * @brief Format a row of cells according to column definitions.
   *
   * Each cell is padded to its column's width using ANSI-safe padding.
   */
  inline std::string format_row(
    const std::vector<Column>& columns,
    const Row&                 cells,
    const std::string&         sep = "  ") {
    std::string line;

    for (std::size_t i = 0; i < columns.size() && i < cells.size(); ++i) {
      if (i > 0) line += sep;

      line += pad(cells[i], columns[i].width, columns[i].align);
    }

    return line;
  }

  /**
   * @brief Generate a separator line spanning the full table width.
   */
  inline std::string format_separator(
    const std::vector<Column>& columns,
    const std::string&         sep = "  ") {
    int total = 0;

    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (i > 0) total += static_cast<int>(sep.size());

      total += columns[i].width;
    }

    return std::string(static_cast<std::size_t>(total), '-');
  }

  /**
   * @brief Extract header labels from column definitions as a Row.
   */
  inline Row header_labels(const std::vector<Column>& columns) {
    Row labels;
    labels.reserve(columns.size());

    for (const auto& col : columns) {
      labels.push_back(col.label);
    }

    return labels;
  }

  /**
   * @brief Format a row as a markdown table row.
   */
  inline std::string format_md_row(const Row& cells) {
    std::string line = "|";

    for (const auto& cell : cells) {
      line += " " + cell + " |";
    }

    return line;
  }

  /**
   * @brief Generate a markdown alignment row.
   */
  inline std::string format_md_separator(const std::vector<Column>& columns) {
    std::string line = "|";

    for (const auto& col : columns) {
      line += (col.align == Align::right) ? "---:|" : ":---|";
    }

    return line;
  }
}
