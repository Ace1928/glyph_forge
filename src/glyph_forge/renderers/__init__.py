"""Minimal rendering utilities for Glyph Forge."""

from __future__ import annotations

import html
from typing import Any, Dict, List

GlyphMatrix = List[List[str]]


class TextRenderer:
    """Render glyph matrix as plain text."""

    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        return "\n".join("".join(row) for row in matrix)


class HTMLRenderer:
    """Render an escaped glyph matrix inside an HTML ``pre`` element."""

    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        del options
        text = html.escape("\n".join("".join(row) for row in matrix))
        return f"<pre style='line-height:1;letter-spacing:0'>{text}</pre>"


class ANSIRenderer(TextRenderer):
    """Plain terminal renderer retained under its historical API name."""

    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        return super().render(matrix, options)


class SVGRenderer:
    """Render glyph matrix to a very basic SVG document."""

    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        char_width = float((options or {}).get("char_width", 10))
        char_height = float((options or {}).get("char_height", 14))
        if char_width <= 0 or char_height <= 0:
            raise ValueError("SVG character dimensions must be positive")
        width = max((len(row) for row in matrix), default=0) * char_width
        height = len(matrix) * char_height
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        ]
        for i, row in enumerate(matrix):
            y = (i + 1) * char_height
            text = html.escape("".join(row))
            svg_lines.append(
                f'<text x="0" y="{y}" font-family="monospace">{text}</text>'
            )
        svg_lines.append("</svg>")
        return "\n".join(svg_lines)


__all__ = ["TextRenderer", "HTMLRenderer", "ANSIRenderer", "SVGRenderer"]
