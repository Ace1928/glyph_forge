"""Focused smoke tests for the optional full-screen Textual interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from glyph_forge.ui.tui import GlyphForgeApp, filter_media_paths  # noqa: E402


def test_media_tree_filters_hidden_and_unknown_files(tmp_path: Path) -> None:
    visible = tmp_path / "photo.png"
    video = tmp_path / "clip.mp4"
    hidden = tmp_path / ".secret.jpg"
    unknown = tmp_path / "notes.md"
    folder = tmp_path / "album"
    for path in (visible, video, hidden, unknown):
        path.write_bytes(b"")
    folder.mkdir()
    filtered = set(filter_media_paths([visible, video, hidden, unknown, folder]))

    assert filtered == {visible, video, folder}


def test_tui_mounts_every_primary_workflow() -> None:
    async def exercise() -> None:
        app = GlyphForgeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#image-tab")
            assert app.query_one("#text-tab")
            assert app.query_one("#live-tab")
            assert app.query_one("#runtime-tab")
            assert app.query_one("#image-browse")
            assert app.query_one("#runtime-studio")

    asyncio.run(exercise())


def test_tui_image_preview_runs_off_the_ui_thread(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "sample.png"
    Image.linear_gradient("L").resize((16, 8)).save(source)

    async def exercise() -> None:
        app = GlyphForgeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#image-path").value = str(source)
            app.query_one("#image-width").value = "8"
            await pilot.click("#image-convert")
            await pilot.pause(0.4)
            assert app.image_result
            assert "Error" not in app.image_result

    asyncio.run(exercise())
