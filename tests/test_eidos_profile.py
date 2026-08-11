"""Tests for the bundled design-profile loader."""

import os
import shutil
from pathlib import Path

from glyph_forge.eidos_profile import (
    BUNDLED_PROFILE_PATH,
    load_profile,
    save_profile,
    update_profile,
)


def test_load_profile():
    profile = load_profile()
    assert "identity" in profile
    assert "psychology" in profile


def test_update_profile(tmp_path: Path):
    temp_profile = tmp_path / "profile.yml"
    shutil.copy(BUNDLED_PROFILE_PATH, temp_profile)

    data = {"values": ["adaptability"]}
    updated = update_profile(data, path=temp_profile)
    reloaded = load_profile(path=temp_profile)

    assert "adaptability" in reloaded["values"]
    assert updated == reloaded


def test_default_profile_is_portable_and_user_writable(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "profile.yml"
    monkeypatch.setenv("GLYPH_FORGE_EIDOS_PROFILE", str(destination))

    bundled = load_profile()
    updated = update_profile({"identity": {"alias": "portable"}})

    assert bundled["identity"]["official_name"] == "Eidos"
    assert updated["identity"]["alias"] == "portable"
    assert destination.is_file()


def test_profile_save_is_atomic_and_private(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "profile.yml"

    save_profile({"values": ["durability"]}, destination)

    assert load_profile(destination)["values"] == ["durability"]
    assert not list(destination.parent.glob(".*.tmp"))
    # Windows inherits the per-user directory ACL and does not expose POSIX
    # group/world permission semantics through ``st_mode``.
    if os.name == "posix" and destination.stat().st_mode & 0o077:
        raise AssertionError("User profile must not be group/world accessible")


def test_legacy_profile_is_read_then_migrated_to_canonical_config_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from glyph_forge import eidos_profile

    monkeypatch.delenv("GLYPH_FORGE_EIDOS_PROFILE", raising=False)
    canonical_root = tmp_path / "canonical"
    monkeypatch.setenv("GLYPH_FORGE_CONFIG_HOME", str(canonical_root))
    legacy = tmp_path / "legacy" / "eidos_profile.yml"
    monkeypatch.setattr(eidos_profile, "_legacy_user_profile_path", lambda: legacy)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("values:\n  - legacy\n", encoding="utf-8")

    assert load_profile()["values"] == ["legacy"]
    update_profile({"values": ["migrated"]})

    canonical = canonical_root / "eidos_profile.yml"
    assert load_profile(canonical)["values"] == ["migrated"]
    assert legacy.read_text(encoding="utf-8") == "values:\n  - legacy\n"
