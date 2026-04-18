"""Shared stdlib-only rendering primitives used by all MASM visualizers."""

from __future__ import annotations


def truncate(text: str, width: int = 55) -> str:
    """Truncate text with ellipsis if it exceeds width."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def ascii_bar(value: float, max_value: float = 1.0, width: int = 30, fill: str = "#") -> str:
    """
    Return an ASCII progress bar string.

    Example: [##########--------------------]
    """
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
        filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{fill * filled}{'-' * empty}]"


def format_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
) -> str:
    """
    Return a left-aligned ASCII table with | separators and a header divider.

    Example:
      ID       | Agent      | Content           | State
      ---------|------------|-------------------|-------
      abc12345 | researcher | Revenue grew 23%  | ACTIVE
    """
    def _row(cells: list[str]) -> str:
        parts = [str(c).ljust(col_widths[i]) for i, c in enumerate(cells)]
        return " | ".join(parts)

    divider = "-+-".join("-" * w for w in col_widths)
    lines = [_row(headers), divider]
    for row in rows:
        lines.append(_row(row))
    return "\n".join(lines)


def section_header(title: str, width: int = 70) -> str:
    """
    Return a section header block.

    Example:
      ======================================================================
      CONFLICT GRAPH
      ======================================================================
    """
    bar = "=" * width
    return f"{bar}\n{title}\n{bar}"


def state_icon(state_value: str) -> str:
    """Map a MemoryState value string to a single-character icon."""
    return {
        "active": "+",
        "superseded": "~",
        "disputed": "!",
        "retracted": "-",
        "forgotten": "X",
        "deprecated": "d",
    }.get(state_value, "?")


def severity_badge(severity_value: str) -> str:
    """Return a fixed-width 8-char severity badge, e.g. [HIGH   ]."""
    label = severity_value.upper()[:8]
    return f"[{label:<8}]"
