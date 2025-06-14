"""
🧿 Eidos Profile API
--------------------

Self-aware loader and updater for the `eidos_profile.yml` file. Follows the
Eidosian principles of precision and exhaustive clarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, TypedDict, cast

import yaml

logger = logging.getLogger(__name__)

# Default path two levels above this file (project root)
PROFILE_PATH = Path(__file__).resolve().parents[2] / "eidos_profile.yml"


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
    """Load Eidos profile from YAML with surgical precision."""
    profile_path = path or PROFILE_PATH
    with open(profile_path, "r", encoding="utf-8") as f:
        data: EidosProfile = yaml.safe_load(f)
    logger.debug("Loaded profile from %s", profile_path)
    return data


def save_profile(profile: EidosProfile, path: Path | None = None) -> None:
    """Persist profile data back to YAML."""
    profile_path = path or PROFILE_PATH
    with open(profile_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False)
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
