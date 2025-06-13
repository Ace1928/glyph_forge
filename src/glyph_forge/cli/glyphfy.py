#!/usr/bin/env python3
"""Compatibility shim for legacy `glyphfy` command.

This module forwards execution to :mod:`glyph_forge.cli.imagize` so that
existing scripts depending on the old ``glyphfy`` entry point continue to
operate without changes.
"""

from .imagize import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
