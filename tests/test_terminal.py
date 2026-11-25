"""Tests for portable terminal capability selection."""

from __future__ import annotations

from io import StringIO

import pytest

from glyph_forge.terminal import TerminalCapabilities, detect_terminal


class FakeStream(StringIO):
    """A small text stream with configurable terminal properties."""

    def __init__(self, *, interactive: bool, encoding: str = "utf-8") -> None:
        super().__init__()
        self._interactive = interactive
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    def isatty(self) -> bool:
        return self._interactive


def test_redirected_output_disables_control_sequences() -> None:
    capabilities = detect_terminal(FakeStream(interactive=False), {"TERM": "xterm-256color"})

    assert capabilities.color_depth == 0
    assert capabilities.resolve_color("truecolor", streaming=True) == "none"
    assert not capabilities.interactive


def test_truecolor_terminal_selects_highest_quality() -> None:
    capabilities = detect_terminal(
        FakeStream(interactive=True),
        {"TERM": "xterm-256color", "COLORTERM": "truecolor"},
    )

    assert capabilities.preferred_color == "truecolor"
    assert capabilities.resolve_color("auto", streaming=True) == "truecolor"


@pytest.mark.parametrize("environment", [{"TERM": "dumb"}, {"GLYPH_FORGE_ASCII": "1"}])
def test_minimal_environments_use_ascii(environment: dict[str, str]) -> None:
    capabilities = detect_terminal(FakeStream(interactive=True), environment)

    assert capabilities.resolve_render_mode("braille") == "ascii"


def test_unsupported_modes_fail_clearly() -> None:
    capabilities = TerminalCapabilities(True, 80, 24, 256, True, False, "xterm", "utf-8")

    with pytest.raises(ValueError, match="unsupported color mode"):
        capabilities.resolve_color("magic")
    with pytest.raises(ValueError, match="unsupported render mode"):
        capabilities.resolve_render_mode("pixels")
