"""Glyph Forge public package.

Importing :mod:`glyph_forge` has no filesystem, logging, signal-handler, or
optional dependency side effects. Larger components load only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

from glyph_forge._version import VERSION, __version__

__author__ = "Lloyd Handyside"
__license__ = "MIT"

_LAZY_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "ANSIRenderer": ("glyph_forge.renderers", "ANSIRenderer"),
    "ColorMode": ("glyph_forge.services.image_to_glyph", "ColorMode"),
    "DitherAlgorithm": ("glyph_forge.services.image_to_glyph", "DitherAlgorithm"),
    "GlyphForgeAPI": ("glyph_forge.api", "GlyphForgeAPI"),
    "GlyphForgeConfigError": ("glyph_forge.exceptions", "GlyphForgeConfigError"),
    "GlyphForgeDependencyError": ("glyph_forge.exceptions", "GlyphForgeDependencyError"),
    "GlyphForgeError": ("glyph_forge.exceptions", "GlyphForgeError"),
    "GlyphForgeInputError": ("glyph_forge.exceptions", "GlyphForgeInputError"),
    "GlyphForgeRenderError": ("glyph_forge.exceptions", "GlyphForgeRenderError"),
    "GlyphRenderer": ("glyph_forge.streaming.core.renderer", "GlyphRenderer"),
    "HTMLRenderer": ("glyph_forge.renderers", "HTMLRenderer"),
    "ImageGlyphConverter": ("glyph_forge.services.image_to_glyph", "ImageGlyphConverter"),
    "RenderConfig": ("glyph_forge.streaming.core.renderer", "RenderConfig"),
    "SVGRenderer": ("glyph_forge.renderers", "SVGRenderer"),
    "TextRenderer": ("glyph_forge.renderers", "TextRenderer"),
    "TerminalCapabilities": ("glyph_forge.terminal", "TerminalCapabilities"),
    "detect_terminal": ("glyph_forge.terminal", "detect_terminal"),
    "get_api": ("glyph_forge.api", "get_api"),
    "image_to_glyph": ("glyph_forge.services.image_to_glyph", "image_to_glyph"),
    "text_to_banner": ("glyph_forge.services.text_to_banner", "text_to_banner"),
    "video_to_glyph_frames": (
        "glyph_forge.services.video_to_glyph",
        "video_to_glyph_frames",
    ),
}


def __getattr__(name: str) -> Any:
    """Load a public component on first access."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and lazy public names for interactive discovery."""

    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "ANSIRenderer",
    "ColorMode",
    "DitherAlgorithm",
    "GlyphForgeAPI",
    "GlyphForgeConfigError",
    "GlyphForgeDependencyError",
    "GlyphForgeError",
    "GlyphForgeInputError",
    "GlyphForgeRenderError",
    "GlyphRenderer",
    "HTMLRenderer",
    "ImageGlyphConverter",
    "RenderConfig",
    "SVGRenderer",
    "TextRenderer",
    "TerminalCapabilities",
    "VERSION",
    "__version__",
    "detect_terminal",
    "get_api",
    "image_to_glyph",
    "text_to_banner",
    "video_to_glyph_frames",
]
