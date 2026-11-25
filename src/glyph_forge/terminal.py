"""Portable terminal capability detection and graceful output selection.

The helpers in this module inspect a stream and its environment without
writing to the terminal. Applications can therefore choose safe defaults for
interactive terminals, redirected output, CI logs, and minimal ``TERM=dumb``
environments before rendering a frame.
"""

from __future__ import annotations

import codecs
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import IO, Mapping


@dataclass(frozen=True, slots=True)
class TerminalCapabilities:
    """A stable snapshot of terminal features relevant to Glyph Forge."""

    interactive: bool
    width: int
    height: int
    color_depth: int
    unicode: bool
    hyperlinks: bool
    term: str
    encoding: str

    @property
    def preferred_color(self) -> str:
        """Return the highest safe color mode for the current stream."""

        if self.color_depth >= 16_777_216:
            return "truecolor"
        if self.color_depth >= 256:
            return "ansi256"
        if self.color_depth >= 16:
            return "ansi16"
        return "none"

    @property
    def preferred_stream_color(self) -> str:
        """Return the best mode supported by the streaming renderer."""

        if self.color_depth >= 16_777_216:
            return "truecolor"
        if self.color_depth >= 256:
            return "ansi256"
        return "none"

    @property
    def preferred_render_mode(self) -> str:
        """Return a Unicode gradient or the portable ASCII fallback."""

        return "gradient" if self.unicode else "ascii"

    def resolve_color(self, requested: str, *, streaming: bool = False) -> str:
        """Resolve a requested color mode, downgrading unsupported output."""

        normalized = requested.lower()
        allowed = {"auto", "ansi", "ansi16", "ansi256", "none", "truecolor"}
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"unsupported color mode {requested!r}; choose one of: {choices}")
        if normalized == "none" or self.color_depth == 0:
            return "none"
        if normalized == "auto":
            return self.preferred_stream_color if streaming else self.preferred_color
        if normalized in {"ansi", "truecolor"} and self.color_depth >= 16_777_216:
            return "truecolor"
        if normalized in {"ansi", "truecolor", "ansi256"} and self.color_depth >= 256:
            return "ansi256"
        if not streaming and self.color_depth >= 16:
            return "ansi16"
        return "none"

    def resolve_render_mode(self, requested: str) -> str:
        """Resolve a rendering mode with an ASCII compatibility fallback."""

        normalized = requested.lower()
        allowed = {"ascii", "auto", "blocks", "braille", "gradient", "hybrid"}
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"unsupported render mode {requested!r}; choose one of: {choices}")
        if normalized == "auto":
            return self.preferred_render_mode
        if not self.unicode and normalized != "ascii":
            return "ascii"
        return normalized

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _supports_unicode(encoding: str, *, forced_ascii: bool) -> bool:
    if forced_ascii:
        return False
    try:
        codecs.lookup(encoding)
        "\u2588\u2800".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _is_interactive(stream: IO[str]) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError):
        return False


def detect_terminal(
    stream: IO[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> TerminalCapabilities:
    """Detect capabilities for ``stream`` using portable, side-effect-free checks."""

    target = stream or sys.stdout
    env = os.environ if environ is None else environ
    term = env.get("TERM", "").lower()
    colorterm = env.get("COLORTERM", "").lower()
    encoding = getattr(target, "encoding", None) or "utf-8"
    interactive = _is_interactive(target)
    forced_color = env.get("FORCE_COLOR", "").lower() not in {"", "0", "false", "no"}
    no_color = "NO_COLOR" in env or term == "dumb"

    if no_color or (not interactive and not forced_color):
        color_depth = 0
    elif "truecolor" in colorterm or "24bit" in colorterm:
        color_depth = 16_777_216
    elif "256color" in term:
        color_depth = 256
    else:
        color_depth = 16

    forced_ascii = env.get("GLYPH_FORGE_ASCII", "").lower() in {"1", "true", "yes", "on"}
    unicode_supported = term != "dumb" and _supports_unicode(
        encoding,
        forced_ascii=forced_ascii,
    )
    size = shutil.get_terminal_size(fallback=(80, 24))
    term_program = env.get("TERM_PROGRAM", "").lower()
    hyperlinks = interactive and (
        "WT_SESSION" in env
        or "VTE_VERSION" in env
        or term_program in {"ghostty", "iterm.app", "vscode", "wezterm"}
    )

    return TerminalCapabilities(
        interactive=interactive,
        width=max(1, size.columns),
        height=max(1, size.lines),
        color_depth=color_depth,
        unicode=unicode_supported,
        hyperlinks=hyperlinks,
        term=term,
        encoding=encoding,
    )


__all__ = ["TerminalCapabilities", "detect_terminal"]
