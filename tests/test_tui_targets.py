"""Targeted TUI tests for run_tui without launching UI."""

from __future__ import annotations

import pytest

pytest.importorskip("textual", reason="textual not installed")


def test_run_tui_noop(monkeypatch):
    from glyph_forge.ui import tui

    class DummyApp:
        def run(self):
            return None

    monkeypatch.setattr(tui, "GlyphForgeApp", DummyApp)
    tui.run_tui()
