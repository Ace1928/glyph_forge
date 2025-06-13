"""Eidosian test suite for the profile loader."""

from pathlib import Path
import tempfile
import shutil

from glyph_forge.eidos_profile import load_profile, update_profile


def test_load_profile():
    profile = load_profile()
    assert "identity" in profile
    assert "psychology" in profile


def test_update_profile(tmp_path: Path):
    temp_profile = tmp_path / "profile.yml"
    shutil.copy(Path(__file__).parents[1] / "eidos_profile.yml", temp_profile)

    data = {"values": ["adaptability"]}
    updated = update_profile(data, path=temp_profile)
    reloaded = load_profile(path=temp_profile)

    assert "adaptability" in reloaded["values"]
    assert updated == reloaded
