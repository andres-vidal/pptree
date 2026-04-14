"""Parse genhtml HTML output and produce a coverage report.

Expects genhtml to have already been run on a filtered .info file.
Parses the resulting HTML for consistent line and function coverage reporting.

Output format:
  - Terminal (not CI): aligned table for human reading.
  - CI (GITHUB_OUTPUT set): writes `line_pct` and `report` (markdown) for
    PR comments and badge updates.
  - CI without GITHUB_OUTPUT: markdown to stdout.
"""
import os
import re
import sys


def parse_html_index(html_path):
    """Parse a genhtml index.html and extract the coverage table rows.

    Returns (total, rows) where:
      - total: {line_pct, line_total, line_hit, fn_pct, fn_total, fn_hit}
      - rows: [{name, line_pct, line_total, line_hit, fn_pct, fn_total, fn_hit}]
    """
    with open(html_path) as f:
        html = f.read()

    # Extract totals from header
    total = {}
    m = re.search(r'headerItem">Lines:.*?headerCovTableEntry\w*">([\d.]+)', html, re.DOTALL)
    if m:
        total["line_pct"] = float(m.group(1))
    m = re.search(r'headerItem">Lines:.*?headerCovTableEntry">(\d+).*?headerCovTableEntry">(\d+)', html, re.DOTALL)
    if m:
        total["line_total"] = int(m.group(1))
        total["line_hit"] = int(m.group(2))
    m = re.search(r'headerItem">Functions:.*?headerCovTableEntry\w*">([\d.]+)', html, re.DOTALL)
    if m:
        total["fn_pct"] = float(m.group(1))
    m = re.search(r'headerItem">Functions:.*?headerCovTableEntry">(\d+).*?headerCovTableEntry">(\d+)', html, re.DOTALL)
    if m:
        total["fn_total"] = int(m.group(1))
        total["fn_hit"] = int(m.group(2))

    # Extract table rows (directories or files)
    rows = []
    row_pattern = re.compile(
        r'cover(?:Directory|File)">'
        r'(?:<a[^>]*>)?([^<]+)(?:</a>)?'
        r'.*?coverPer\w+">([\d.]+)'
        r'.*?coverNumDflt">(\d+)'
        r'.*?coverNumDflt">(\d+)'
        r'.*?coverPer\w+">([\d.]+)'
        r'.*?coverNumDflt">(\d+)'
        r'.*?coverNumDflt">(\d+)',
        re.DOTALL,
    )

    for m in row_pattern.finditer(html):
        rows.append({
            "name": m.group(1).rstrip("/"),
            "line_pct": float(m.group(2)),
            "line_total": int(m.group(3)),
            "line_hit": int(m.group(4)),
            "fn_pct": float(m.group(5)),
            "fn_total": int(m.group(6)),
            "fn_hit": int(m.group(7)),
        })

    return total, rows


def parse_html_all(html_dir):
    """Parse the top-level and all subdirectory index pages.

    Returns (total, modules, file_entries) where:
      - total: {line_pct, line_total, line_hit, fn_pct, fn_total, fn_hit}
      - modules: [{name, line_pct, ...}] sorted by line_pct
      - file_entries: [{name, line_pct, ...}] sorted by line_pct
    """
    total, modules = parse_html_index(os.path.join(html_dir, "index.html"))
    modules.sort(key=lambda r: r["line_pct"])

    file_entries = []
    for mod in modules:
        mod_dir = os.path.join(html_dir, mod["name"])
        if os.path.isdir(mod_dir):
            index = os.path.join(mod_dir, "index.html")
            if os.path.exists(index):
                _, files = parse_html_index(index)
                for f in files:
                    f["name"] = mod["name"] + "/" + f["name"]
                file_entries.extend(files)

    file_entries.sort(key=lambda r: (r["line_pct"], r["name"]))
    return total, modules, file_entries


def format_terminal(total, modules, file_entries):
    """Build a terminal-friendly coverage report."""
    lines = []

    # Per-file detail
    lines.append("")
    lines.append("  {:<50} {:>10} {:>10}".format("File", "Lines", "Functions"))
    lines.append("  " + "-" * 72)

    for entry in file_entries:
        lines.append("  {:<50} {:>9.1f}% {:>9.1f}%".format(
            entry["name"], entry["line_pct"], entry["fn_pct"]))

    # Module summary
    lines.append("")
    lines.append("  {:<50} {:>10} {:>10}".format("Module", "Lines", "Functions"))
    lines.append("  " + "-" * 72)

    for mod in modules:
        lines.append("  {:<50} {:>9.1f}% {:>9.1f}%".format(
            mod["name"], mod["line_pct"], mod["fn_pct"]))

    lines.append("  " + "-" * 72)
    lines.append("  {:<50} {:>9.1f}% {:>9.1f}%".format(
        "Total", total["line_pct"], total["fn_pct"]))
    lines.append("")

    return "\n".join(lines)


def format_markdown(total, modules, file_entries):
    """Build a markdown coverage report for CI."""
    lines = []
    lines.append("## Code Coverage")
    lines.append("")

    # Per-file detail in collapsible section
    lines.append("<details>")
    lines.append("<summary>Per-file breakdown</summary>")
    lines.append("")
    lines.append("| File | Lines | Functions |")
    lines.append("|------|------:|----------:|")

    for entry in file_entries:
        lines.append("| {} | {:.1f}% | {:.1f}% |".format(
            entry["name"], entry["line_pct"], entry["fn_pct"]))

    lines.append("")
    lines.append("</details>")
    lines.append("")

    # Module summary
    lines.append("| Module | Lines | Functions |")
    lines.append("|--------|------:|----------:|")

    for mod in modules:
        lines.append("| {} | {:.1f}% | {:.1f}% |".format(
            mod["name"], mod["line_pct"], mod["fn_pct"]))

    lines.append("| **Total** | **{:.1f}%** | **{:.1f}%** |".format(
        total["line_pct"], total["fn_pct"]))

    return "\n".join(lines)


def main():
    html_dir = sys.argv[1] if len(sys.argv) > 1 else ".coverage/html"

    total, modules, file_entries = parse_html_all(html_dir)

    gh_output = os.environ.get("GITHUB_OUTPUT")

    if gh_output:
        report = format_markdown(total, modules, file_entries)
        with open(gh_output, "a") as out:
            out.write("line_pct={:.1f}\n".format(total["line_pct"]))
            out.write("report<<EOF\n")
            out.write(report + "\n")
            out.write("EOF\n")
    elif os.environ.get("CI"):
        print(format_markdown(total, modules, file_entries))
    else:
        print(format_terminal(total, modules, file_entries))


if __name__ == "__main__":
    main()
