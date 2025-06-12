"""Eidosian tests for border detection and removal utilities."""

import pytest

from glyph_forge.core.style_manager import (
    apply_style,
    detect_border_style,
    remove_border,
)


class TestBorderUtilities:
    """Atomic verification of border detection and stripping."""

    def test_detect_border_style(self) -> None:
        """Detect border style applied by ``apply_style``."""
        art = apply_style("Forge", style_name="boxed", padding=0)

        assert detect_border_style(art) == "single"

    def test_remove_border(self) -> None:
        """Remove border and recover original text."""
        original = "Glyph Forge"
        art = apply_style(original, style_name="boxed", padding=0)

        stripped = remove_border(art)

        assert stripped.strip() == original

