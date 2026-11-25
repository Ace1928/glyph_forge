"""Single source of truth for the installed Glyph Forge version."""

VERSION = (1, 0, 0)
__version__ = ".".join(str(part) for part in VERSION)

__all__ = ["VERSION", "__version__"]
