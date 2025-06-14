"""Minimal rendering utilities for Glyph Forge."""
from __future__ import annotations
from typing import List, Dict, Any

GlyphMatrix = List[List[str]]

class TextRenderer:
    """Render glyph matrix as plain text."""
    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        return "\n".join("".join(row) for row in matrix)

class HTMLRenderer:
    """Render glyph matrix inside simple HTML <pre> block."""
    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        lines = ["<pre style='line-height:1; letter-spacing:0'>"]
        for row in matrix:
            lines.append("".join(row))
            lines.append("<br>")
        lines.append("</pre>")
        return "".join(lines)

class ANSIRenderer(TextRenderer):
    """Alias to :class:`TextRenderer` for backward compatibility."""
    pass

class SVGRenderer:
    """Render glyph matrix to a very basic SVG document."""
    def render(self, matrix: GlyphMatrix, options: Dict[str, Any] | None = None) -> str:
        char_width = (options or {}).get("char_width", 10)
        char_height = (options or {}).get("char_height", 14)
        width = max(len(row) for row in matrix) * char_width
        height = len(matrix) * char_height
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        ]
        for i, row in enumerate(matrix):
            y = (i + 1) * char_height
            text = "".join(row).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            svg_lines.append(f'<text x="0" y="{y}" font-family="monospace">{text}</text>')
        svg_lines.append("</svg>")
        return "\n".join(svg_lines)

__all__ = ["TextRenderer", "HTMLRenderer", "ANSIRenderer", "SVGRenderer"]
