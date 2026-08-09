"""Tests for the bundled design-profile loader."""

import shutil
from pathlib import Path

from glyph_forge.eidos_profile import (
    BUNDLED_PROFILE_PATH,
    load_profile,
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
