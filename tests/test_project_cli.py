"""End-to-end CLI coverage for projects, presets, recovery, and batches."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.contracts import RenderRequest
from glyph_forge.projects import ProjectSession, load_project, recovery_path

runner = CliRunner()


def source_image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4), color).save(path)


def invoke(tmp_path: Path, arguments: list[str]):
    return runner.invoke(
        app,
        arguments,
        env={"GLYPH_FORGE_CONFIG_HOME": str(tmp_path / "config")},
    )


def test_cli_creates_portable_project_and_round_trips_preset_and_variant(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outside" / "photo.png"
    source_image(source)
    project = tmp_path / "workspace" / "art.glyphforge.json"

    created = invoke(
        tmp_path,
        ["project", "new", str(project), str(source), "--name", "Art"],
    )

    assert created.exit_code == 0, created.output
    document = load_project(project)
    assert document.name == "Art"
    assert document.source.path == "assets/photo.png"
    assert document.source.resolve(project).read_bytes() == source.read_bytes()

    info = invoke(tmp_path, ["project", "info", str(project), "--json"])
    assert info.exit_code == 0, info.output
    assert json.loads(info.stdout)["source_available"] is True

    preset = tmp_path / "shared.glyphpreset.json"
    exported = invoke(
        tmp_path,
        ["preset", "export", str(project), str(preset), "--name", "Shared"],
    )
    assert exported.exit_code == 0, exported.output
    applied = invoke(
        tmp_path,
        [
            "preset",
            "apply",
            str(preset),
            str(project),
            "--new-variant",
            "shared",
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert [item.identifier for item in load_project(project).variants] == [
        "default",
        "shared",
    ]

    selected = invoke(tmp_path, ["project", "variant-select", str(project), "default"])
    removed = invoke(tmp_path, ["project", "variant-remove", str(project), "shared"])
    assert selected.exit_code == removed.exit_code == 0
    assert len(load_project(project).variants) == 1


def test_cli_project_render_and_recovery_are_machine_readable(tmp_path: Path) -> None:
    project_dir = tmp_path / "workspace"
    source = project_dir / "source.png"
    source_image(source)
    project = project_dir / "art.glyphforge.json"
    assert (
        invoke(tmp_path, ["project", "new", str(project), str(source)]).exit_code == 0
    )
    interrupted = ProjectSession.open(project, autosave_delay=0)
    interrupted.update_active_request(RenderRequest(width=7, height=3))
    assert recovery_path(project).is_file()

    recovered = invoke(tmp_path, ["project", "recover", str(project)])
    assert recovered.exit_code == 0, recovered.output
    assert load_project(project).active.request.width == 7
    assert not recovery_path(project).exists()

    output = tmp_path / "render.txt"
    rendered = invoke(
        tmp_path,
        [
            "project",
            "render",
            str(project),
            "--output",
            str(output),
            "--json",
        ],
    )
    assert rendered.exit_code == 0, rendered.output
    metrics = json.loads(rendered.stdout)
    assert metrics["metrics"]["columns"] == 7
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_preset_create_and_batch_share_one_exact_request(tmp_path: Path) -> None:
    preset = tmp_path / "poster.glyphpreset.json"
    created = invoke(
        tmp_path,
        [
            "preset",
            "create",
            "Poster",
            str(preset),
            "--width",
            "8",
            "--height",
            "4",
            "--format",
            "svg",
            "--output-width",
            "640",
            "--output-height",
            "360",
            "--mode",
            "braille",
        ],
    )
    assert created.exit_code == 0, created.output
    first = tmp_path / "one" / "same.png"
    second = tmp_path / "two" / "same.png"
    source_image(first, "black")
    source_image(second, "white")
    output = tmp_path / "batch"

    batch = invoke(
        tmp_path,
        [
            "batch",
            str(preset),
            str(first),
            str(second),
            "--output-dir",
            str(output),
            "--workers",
            "2",
            "--json",
        ],
    )

    assert batch.exit_code == 0, batch.output
    report = json.loads(batch.stdout)
    assert report["succeeded"] == 2
    assert sorted(path.name for path in output.iterdir()) == [
        "same-2.glyph.svg",
        "same.glyph.svg",
    ]
    assert all(
        'viewBox="0 0 640.00 360.00"' in path.read_text(encoding="utf-8")
        for path in output.iterdir()
    )


def test_reference_only_rejects_external_media_without_leaving_a_project(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outside.png"
    source_image(source)
    project = tmp_path / "workspace" / "art.glyphforge.json"

    result = invoke(
        tmp_path,
        ["project", "new", str(project), str(source), "--reference-only"],
    )

    assert result.exit_code == 2
    assert "inside the project directory" in result.output
    assert not project.exists()
