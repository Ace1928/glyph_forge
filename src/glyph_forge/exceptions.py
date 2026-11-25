"""Public exception hierarchy for predictable Glyph Forge error handling."""


class GlyphForgeError(Exception):
    """Base class for errors that callers may handle uniformly."""


class GlyphForgeInputError(GlyphForgeError, ValueError):
    """Raised when an input or rendering option is invalid."""


class GlyphForgeRenderError(GlyphForgeError):
    """Raised when valid input cannot be decoded, rendered, or written."""


class GlyphForgeDependencyError(GlyphForgeError, ImportError):
    """Raised when a requested optional feature is not installed."""


class GlyphForgeConfigError(GlyphForgeError):
    """Raised when configuration cannot be validated or persisted."""


__all__ = [
    "GlyphForgeDependencyError",
    "GlyphForgeConfigError",
    "GlyphForgeError",
    "GlyphForgeInputError",
    "GlyphForgeRenderError",
]
