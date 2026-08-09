"""PyInstaller entry point for the portable Glyph Forge bundle."""

from __future__ import annotations

from multiprocessing import freeze_support

from glyph_forge.cli import main

if __name__ == "__main__":
    freeze_support()
    main()
