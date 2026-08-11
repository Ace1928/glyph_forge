"""Focused smoke tests for the optional full-screen Textual interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from textual.containers import VerticalScroll  # noqa: E402

from glyph_forge.ui.tui import (  # noqa: E402
    DocumentDirectoryTree,
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


def test_document_tree_filters_to_projects_and_presets(tmp_path: Path) -> None:
    project = tmp_path / "art.glyphforge.json"
    preset = tmp_path / "look.glyphpreset.json"
    unrelated = tmp_path / "data.json"
    folder = tmp_path / "folder"
    for path in (project, preset, unrelated):
        path.write_text("{}", encoding="utf-8")
    folder.mkdir()
    tree = object.__new__(DocumentDirectoryTree)

    assert set(tree.filter_paths([project, preset, unrelated, folder])) == {
        project,
        preset,
        folder,
    }


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
            assert app.query_one("#project-tab")
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


def test_tui_project_variants_presets_and_recovery_use_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    monkeypatch.setenv("GLYPH_FORGE_CONFIG_HOME", str(tmp_path / "config"))
    source = tmp_path / "source.png"
    project = tmp_path / "work" / "art.glyphforge.json"
    preset = tmp_path / "bright.glyphpreset.json"
    Image.linear_gradient("L").resize((16, 8)).save(source)

    async def exercise() -> None:
        app = GlyphForgeApp()
        async with app.run_test(size=(130, 44)) as pilot:
            app.query_one("#image-path").value = str(source)
            app.query_one("#image-width").value = "8"
            app.request_image_conversion()
            await pilot.pause(0.4)
            assert app.image_result

            app.query_one("#project-path").value = str(project)
            app.create_project()
            await pilot.pause(0.4)
            assert app._project_session is not None
            assert app._project_session.project.source.path == "assets/source.png"
            assert not app.query_one("#project-recent").disabled
            assert app.query_one("#project-recent").value == str(project.resolve())

            app.query_one("#project-variant-name").value = "Bright look"
            app.add_project_variant()
            assert app._project_session.project.active_variant == "bright-look"
            app.query_one("#preset-path").value = str(preset)
            app.export_preset()
            assert preset.is_file()

            app.query_one("#image-brightness").value = "1.4"
            app.request_image_conversion()
            await pilot.pause(0.4)
            assert app._project_session.project.active.request.brightness == 1.4
            assert app._project_session.can_undo
            app.action_undo()
            assert app._project_session.project.active.request.brightness != 1.4
            app.action_redo()
            assert app._project_session.project.active.request.brightness == 1.4
            app.save_project()
            assert not app._project_session.dirty

    asyncio.run(exercise())
    assert project.is_file()


def test_tui_runs_a_bounded_batch_from_the_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    monkeypatch.setenv("GLYPH_FORGE_CONFIG_HOME", str(tmp_path / "config"))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    project = tmp_path / "work" / "batch.glyphforge.json"
    output = tmp_path / "outputs"
    Image.new("RGB", (8, 4), "black").save(first)
    Image.new("RGB", (8, 4), "white").save(second)

    async def exercise() -> None:
        app = GlyphForgeApp()
        async with app.run_test(size=(130, 44)) as pilot:
            app.query_one("#image-path").value = str(first)
            app.request_image_conversion()
            await pilot.pause(0.4)
            app.query_one("#project-path").value = str(project)
            app.create_project()
            await pilot.pause(0.3)
            app.add_batch_source()
            app._image_source = str(second)
            app.add_batch_source()
            app.query_one("#batch-output").value = str(output)
            app.query_one("#batch-workers").value = "2"
            app.start_batch()
            for _ in range(40):
                if app._batch_cancellation is None:
                    break
                await pilot.pause(0.05)
            assert app._batch_cancellation is None

    asyncio.run(exercise())
    assert len(list(output.glob("*.txt"))) == 2
