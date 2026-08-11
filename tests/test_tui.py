"""Focused smoke tests for the optional full-screen Textual interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from textual.containers import VerticalScroll  # noqa: E402

from glyph_forge.ui.tui import (  # noqa: E402
    GlyphForgeApp,
    filter_media_paths,
    parse_pixel_dimensions,
)


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


@pytest.mark.parametrize("value", ["1920x1080", "1920×1080", " 333 x 211 "])
def test_tui_pixel_dimensions_are_easy_to_enter(value: str) -> None:
    expected = (333, 211) if "333" in value else (1920, 1080)
    assert parse_pixel_dimensions(value) == expected


@pytest.mark.parametrize("value", ["1080p", "0x720", "9000x720"])
def test_tui_pixel_dimensions_are_bounded(value: str) -> None:
    with pytest.raises(ValueError, match="Output pixels"):
        parse_pixel_dimensions(value)


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


def test_tui_saves_an_exact_size_png_from_the_preview(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "sample.png"
    output = tmp_path / "result.png"
    Image.linear_gradient("L").resize((16, 8)).save(source)

    async def exercise() -> None:
        app = GlyphForgeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#image-path").value = str(source)
            app.query_one("#image-width").value = "8"
            await pilot.click("#image-convert")
            await pilot.pause(0.4)
            assert app.image_result
            app.query_one("#image-output-size").value = "333x211"
            app.query_one("#image-output-path").value = str(output)
            app.query(".form-panel").first(VerticalScroll).scroll_end(animate=False)
            await pilot.pause()
            await pilot.click("#image-save")
            for _ in range(20):
                if output.is_file():
                    break
                await pilot.pause(0.05)

    asyncio.run(exercise())
    with Image.open(output) as exported:
        assert exported.size == (333, 211)
