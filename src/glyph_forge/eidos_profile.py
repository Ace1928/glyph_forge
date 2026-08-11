"""Bundled design-profile metadata with portable user overrides."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, TypedDict, cast

import yaml

from .config.settings import user_config_directory
from .persistence import atomic_write_text

logger = logging.getLogger(__name__)

BUNDLED_PROFILE_PATH = Path(__file__).with_name("resources") / "eidos_profile.yml"


def _user_profile_path() -> Path:
    override = os.environ.get("GLYPH_FORGE_EIDOS_PROFILE")
    if override:
        return Path(override).expanduser()
    return user_config_directory() / "eidos_profile.yml"


def _legacy_user_profile_path() -> Path:
    """Return the pre-0.4 path retained as a read-only migration source."""

    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "GlyphForge" / "eidos_profile.yml"
    config_root = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ).expanduser()
    return config_root / "glyph-forge" / "eidos_profile.yml"


PROFILE_PATH = BUNDLED_PROFILE_PATH


class BigFive(TypedDict):
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float


class Psychology(TypedDict):
    mbti: str
    big_five: BigFive
    cognitive_style: str
    creativity: float


class Identity(TypedDict):
    official_name: str
    alias: str
    motto: str
    tagline: str


class EidosProfile(TypedDict, total=False):
    identity: Identity
    psychology: Psychology
    values: list[str]
    motivations: list[str]
    humor_style: str


def load_profile(path: Path | None = None) -> EidosProfile:
    """Load an explicit profile, a user override, or the bundled default."""

    user_path = _user_profile_path()
    legacy_path = _legacy_user_profile_path()
    profile_path = path or (
        user_path
        if user_path.is_file()
        else legacy_path
        if legacy_path != user_path and legacy_path.is_file()
        else PROFILE_PATH
    )
    with open(profile_path, "r", encoding="utf-8") as f:
        data: EidosProfile = yaml.safe_load(f)
    logger.debug("Loaded profile from %s", profile_path)
    return data


def save_profile(profile: EidosProfile, path: Path | None = None) -> None:
    """Persist profile data to an explicit path or portable user config."""

    profile_path = path or _user_profile_path()
    document = yaml.safe_dump(profile, sort_keys=False)
    atomic_write_text(profile_path, document, permissions=0o600)
    logger.debug("Saved profile to %s", profile_path)


def update_profile(updates: Dict[str, Any], path: Path | None = None) -> EidosProfile:
    """Merge updates into the profile and save the result."""
    profile = load_profile(path)
    _merge_dict(cast(Dict[str, Any], profile), updates)
    save_profile(profile, path)
    return profile


def _merge_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Recursively merge overlay into base."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value


__all__ = [
    "BUNDLED_PROFILE_PATH",
    "EidosProfile",
    "PROFILE_PATH",
    "load_profile",
    "save_profile",
    "update_profile",
]
