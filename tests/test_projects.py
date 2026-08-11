"""Portable project, preset, history, and recovery contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyph_forge.contracts import RenderRequest
from glyph_forge.persistence import AtomicWriteError
from glyph_forge.projects import (
    MAX_HISTORY,
    AssetReference,
    GlyphProject,
    ProjectError,
    ProjectRecoveryError,
    ProjectSession,
    ProjectValidationError,
    RecentProjectStore,
    RenderPreset,
    create_portable_project,
    load_preset,
    load_project,
    recovery_path,
    save_preset,
    save_project,
)

NOW = "2026-08-11T10:00:00.000Z"


def project_at(tmp_path: Path, *, width: int = 100) -> tuple[Path, GlyphProject]:
    asset = tmp_path / "assets" / "source.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"not decoded by the project layer")
    path = tmp_path / "example.glyphforge.json"
    project = GlyphProject.create(
        "Example",
        AssetReference.from_path(asset, path),
        RenderRequest(width=width),
        now=NOW,
    )
    return path, project


def test_project_round_trips_as_stable_bounded_json(tmp_path: Path) -> None:
    path, project = project_at(tmp_path)

    assert save_project(project, path) == path
    restored = load_project(path)

    assert restored == project
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "glyph-forge-project"
    assert payload["schema_version"] == 1
    assert payload["source"] == {"kind": "image", "path": "assets/source.png"}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(".*.tmp"))


def test_project_contract_fixture_matches_the_browser_runtime(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "project-contract-v1.json"
    destination = tmp_path / "fixture.glyphforge.json"
    destination.write_bytes(fixture.read_bytes())

    project = load_project(destination)

    assert project.to_dict() == json.loads(fixture.read_text(encoding="utf-8"))


def test_portable_project_creation_copies_external_media_once(tmp_path: Path) -> None:
    source = tmp_path / "outside" / "CON.png"
    source.parent.mkdir()
    source.write_bytes(b"source bytes")
    destination = tmp_path / "work" / "art.glyphforge.json"

    project = create_portable_project(destination, source)

    assert project.source.path == "assets/_CON.png"
    assert project.source.resolve(destination).read_bytes() == b"source bytes"
    assert load_project(destination) == project


def test_portable_project_creation_wraps_copy_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    destination = tmp_path / "work" / "art.glyphforge.json"

    def fail_copy(source_path: Path, destination_path: Path) -> Path:
        raise AtomicWriteError("storage unavailable")

    monkeypatch.setattr("glyph_forge.projects.atomic_copy_file", fail_copy)

    with pytest.raises(ProjectError, match="storage unavailable"):
        create_portable_project(destination, source)
    assert not destination.exists()


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.png",
        "../escape.png",
        "assets/../escape.png",
        "C:/drive.png",
        "a\\b",
        "assets/CON.png",
        "assets/trailing. ",
        "assets/a?.png",
    ],
)
def test_asset_references_reject_nonportable_or_escaping_paths(path: str) -> None:
    with pytest.raises(ProjectValidationError, match="asset path"):
        AssetReference(path)


def test_asset_reference_factory_requires_project_containment(tmp_path: Path) -> None:
    project = tmp_path / "project" / "test.glyphforge.json"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")

    with pytest.raises(ProjectValidationError, match="inside the project directory"):
        AssetReference.from_path(outside, project)


def test_project_load_does_not_require_asset_but_keeps_safe_resolution(
    tmp_path: Path,
) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)
    project.source.resolve(path).unlink()

    restored = load_project(path)

    assert restored.source.resolve(path) == tmp_path / "assets" / "source.png"


def test_preset_round_trip_preserves_the_canonical_render_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cinema.glyphpreset.json"
    preset = RenderPreset(
        "Cinema",
        RenderRequest(
            width=240,
            height=135,
            mode="quadrant",
            output_format="svg",
            output_width=3840,
            output_height=2160,
            fit="cover",
            brightness=1.2,
            contrast=1.1,
        ),
        {"author": "Glyph Forge"},
    )

    save_preset(preset, path)

    assert load_preset(path) == preset
    assert json.loads(path.read_text(encoding="utf-8"))["request"] == (
        preset.request.to_dict()
    )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"schema_version": 99}, "schema version"),
        ({"schema": "other"}, "Not a"),
        ({"surprise": True}, "unknown fields"),
    ],
)
def test_project_rejects_incompatible_or_ambiguous_documents(
    tmp_path: Path,
    update: dict[str, object],
    message: str,
) -> None:
    path, project = project_at(tmp_path)
    payload = project.to_dict() | update
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectValidationError, match=message):
        load_project(path)


def test_project_wraps_invalid_nested_render_contract(tmp_path: Path) -> None:
    path, project = project_at(tmp_path)
    payload = project.to_dict()
    payload["variants"][0]["request"]["width"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectValidationError, match="Invalid variant request"):
        load_project(path)


def test_session_variants_are_non_destructive_and_fully_undoable(
    tmp_path: Path,
) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)
    session = ProjectSession.open(path, autosave_delay=None)

    session.add_variant("bright", "Bright")
    session.update_active_request(RenderRequest(width=200, brightness=1.4))

    assert session.project.active_variant == "bright"
    assert session.project.active.request.width == 200
    assert session.project.variants[0].request.width == 100
    session.undo()
    assert session.project.active.request.width == 100
    session.undo()
    assert len(session.project.variants) == 1
    session.redo()
    session.redo()
    assert session.project.active.request.width == 200
    session.remove_variant("bright")
    assert session.project.active_variant == "default"


def test_session_history_is_bounded_and_new_edits_clear_redo(tmp_path: Path) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)
    session = ProjectSession.open(path, history_limit=3, autosave_delay=None)
    for width in range(101, 106):
        session.update_active_request(RenderRequest(width=width))

    for _ in range(3):
        session.undo()
    with pytest.raises(ProjectValidationError, match="nothing to undo"):
        session.undo()
    session.redo()
    session.update_active_request(RenderRequest(width=250))
    assert not session.can_redo


def test_immediate_autosave_recovers_edits_after_interruption(tmp_path: Path) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)
    interrupted = ProjectSession.open(path, autosave_delay=0)
    interrupted.update_active_request(RenderRequest(width=333))

    assert recovery_path(path).is_file()
    recovered = ProjectSession.open(path, recover=True, autosave_delay=None)

    assert recovered.dirty
    assert recovered.project.active.request.width == 333
    recovered.save()
    assert load_project(path).active.request.width == 333
    assert not recovery_path(path).exists()


def test_recovery_refuses_to_overwrite_an_externally_changed_project(
    tmp_path: Path,
) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)
    interrupted = ProjectSession.open(path, autosave_delay=0)
    interrupted.update_active_request(RenderRequest(width=333))
    changed = project.with_updates(
        variants=(
            project.active.__class__("default", "Default", RenderRequest(width=444)),
        ),
        now="2026-08-11T10:01:00.000Z",
    )
    save_project(changed, path)

    with pytest.raises(ProjectRecoveryError, match="stale"):
        ProjectSession.open(path, recover=True)
    assert ProjectSession.open(path, recover=False).project.active.request.width == 444


def test_context_manager_checkpoints_unsaved_edits(tmp_path: Path) -> None:
    path, project = project_at(tmp_path)
    save_project(project, path)

    with ProjectSession.open(path, autosave_delay=None) as session:
        session.update_active_request(RenderRequest(width=222))

    assert recovery_path(path).is_file()


def test_recent_projects_are_deduplicated_bounded_and_prunable(tmp_path: Path) -> None:
    clock_values = iter(
        [
            "2026-08-11T10:00:00.000Z",
            "2026-08-11T10:01:00.000Z",
            "2026-08-11T10:02:00.000Z",
            "2026-08-11T10:03:00.000Z",
        ]
    )
    store = RecentProjectStore(
        tmp_path / "recents.json", limit=2, clock=lambda: next(clock_values)
    )
    first = tmp_path / "first.glyphforge.json"
    second = tmp_path / "second.glyphforge.json"
    third = tmp_path / "third.glyphforge.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    third.write_text("{}", encoding="utf-8")

    store.touch(first)
    store.touch(second)
    store.touch(first)
    assert [item.path for item in store.list()] == [first, second]
    store.touch(third)
    assert [item.path for item in store.list()] == [third, first]
    first.unlink()
    store.prune()
    assert [item.path for item in store.list()] == [third]


def test_session_defaults_keep_history_and_recents_explicitly_bounded() -> None:
    assert MAX_HISTORY == 100
