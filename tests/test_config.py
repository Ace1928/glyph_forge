"""Tests for layered, durable configuration behavior."""

from __future__ import annotations

import json

import pytest

from glyph_forge.config.settings import ConfigManager, ConfigScope
from glyph_forge.exceptions import GlyphForgeConfigError, GlyphForgeInputError


def test_reads_do_not_create_user_state(tmp_path) -> None:
    path = tmp_path / "nested" / "config.json"

    manager = ConfigManager(user_path=path, system_path=None)

    assert manager.get("image", "default_width") == 100
    assert not path.parent.exists()


def test_layers_resolve_in_documented_order(tmp_path) -> None:
    system_path = tmp_path / "system.json"
    user_path = tmp_path / "user.json"
    system_path.write_text('{"image": {"default_width": 120}}', encoding="utf-8")
    user_path.write_text('{"image": {"default_width": 140}}', encoding="utf-8")
    manager = ConfigManager(user_path=user_path, system_path=system_path)

    manager.set("image", "default_width", 160, scope=ConfigScope.RUNTIME)

    assert manager.get("image", "default_width") == 160
    assert json.loads(user_path.read_text(encoding="utf-8"))["image"]["default_width"] == 140


def test_user_write_is_minimal_and_persistent(tmp_path) -> None:
    user_path = tmp_path / "config.json"
    manager = ConfigManager(user_path=user_path, system_path=None)

    manager.set("image", "default_width", 72)

    assert json.loads(user_path.read_text(encoding="utf-8")) == {"image": {"default_width": 72}}
    assert ConfigManager(user_path=user_path, system_path=None).get("image", "default_width") == 72


def test_system_scope_and_non_json_values_are_rejected(tmp_path) -> None:
    manager = ConfigManager(user_path=tmp_path / "config.json", system_path=None)

    with pytest.raises(GlyphForgeConfigError, match="read-only"):
        manager.set("image", "default_width", 70, scope=ConfigScope.SYSTEM)
    with pytest.raises(GlyphForgeInputError, match="finite JSON"):
        manager.set("image", "gamma", float("nan"))


def test_reset_removes_overrides_but_preserves_system_layer(tmp_path) -> None:
    system_path = tmp_path / "system.json"
    system_path.write_text('{"image": {"default_width": 120}}', encoding="utf-8")
    manager = ConfigManager(user_path=tmp_path / "user.json", system_path=system_path)
    manager.set("image", "default_width", 80)

    manager.reset_to_defaults("image")

    assert manager.get("image", "default_width") == 120
