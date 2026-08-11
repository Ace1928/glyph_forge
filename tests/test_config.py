"""Production persistence and migration tests for user configuration."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from glyph_forge.config.settings import (
    CONFIG_SCHEMA_VERSION,
    ConfigManager,
    ConfigPersistenceError,
    ConfigScope,
    ConfigValidationError,
    default_user_config_path,
)


def test_explicit_config_home_is_respected_without_eager_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "custom"
    monkeypatch.setenv("GLYPH_FORGE_CONFIG_HOME", str(root))

    path = default_user_config_path()
    manager = ConfigManager()

    assert path == root / "user_config.json"
    assert manager.user_path == path
    assert not root.exists()


def test_legacy_unversioned_file_migrates_on_the_next_write(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"image": {"default_width": 144}}),
        encoding="utf-8",
    )
    manager = ConfigManager(user_path=path)

    assert manager.get("image", "default_width") == 144
    manager.set("image", "brightness", 1.25)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == CONFIG_SCHEMA_VERSION
    assert persisted["settings"] == {
        "image": {"brightness": 1.25, "default_width": 144}
    }


def test_legacy_platform_path_migrates_without_deleting_recovery_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glyph_forge.config import settings

    canonical = tmp_path / "canonical" / "user_config.json"
    legacy = tmp_path / "legacy" / "user_config.json"
    legacy.parent.mkdir()
    original = json.dumps({"image": {"default_width": 144}})
    legacy.write_text(original, encoding="utf-8")
    monkeypatch.delenv("GLYPH_FORGE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("GLYPH_FORGE_CONFIG_HOME", raising=False)
    monkeypatch.setattr(settings, "default_user_config_path", lambda: canonical)
    monkeypatch.setattr(settings, "_legacy_user_config_path", lambda: legacy)

    manager = ConfigManager()

    assert manager.user_path == canonical
    assert manager.get("image", "default_width") == 144
    assert not canonical.exists()
    manager.set("image", "brightness", 1.25)

    persisted = json.loads(canonical.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == CONFIG_SCHEMA_VERSION
    assert persisted["settings"]["image"] == {
        "brightness": 1.25,
        "default_width": 144,
    }
    assert legacy.read_text(encoding="utf-8") == original


def test_explicit_user_path_never_loads_an_implicit_legacy_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glyph_forge.config import settings

    explicit = tmp_path / "explicit.json"
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"image": {"default_width": 144}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "_legacy_user_config_path", lambda: legacy)

    manager = ConfigManager(user_path=explicit)

    assert manager.get("image", "default_width") == 100
    assert not explicit.exists()


def test_invalid_known_values_fail_before_state_or_disk_changes(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    manager = ConfigManager(user_path=path)

    with pytest.raises(ConfigValidationError, match="default_width"):
        manager.set("image", "default_width", 0)
    with pytest.raises(ConfigValidationError, match="brightness"):
        manager.set("image", "brightness", float("nan"))
    with pytest.raises(ConfigValidationError, match="true or false"):
        manager.set("image", "dithering", "yes")

    assert manager.get("image", "default_width") == 100
    assert not path.exists()


def test_runtime_overrides_do_not_persist_and_system_scope_is_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    manager = ConfigManager(user_path=path)

    manager.set("image", "default_width", 72, scope=ConfigScope.RUNTIME)

    assert manager.get("image", "default_width") == 72
    assert not path.exists()
    with pytest.raises(ConfigValidationError, match="read-only"):
        manager.set("image", "default_width", 80, scope=ConfigScope.SYSTEM)


def test_user_write_is_atomic_private_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"
    manager = ConfigManager(user_path=path)

    manager.set("banner", "default_font", "small")

    assert path.is_file()
    assert list(path.parent.glob(".*.tmp")) == []
    assert ConfigManager(user_path=path).get("banner", "default_font") == "small"
    if path.stat().st_mode & 0o077:
        pytest.fail("User configuration must not be group/world accessible")


def test_failed_replace_does_not_leave_a_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glyph_forge import persistence

    path = tmp_path / "settings.json"
    path.write_text("original", encoding="utf-8")
    manager = ConfigManager(user_path=path)
    monkeypatch.setattr(
        persistence.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(ConfigPersistenceError, match="boom"):
        manager.set("banner", "default_font", "small")

    assert path.read_text(encoding="utf-8") == "original"
    assert manager.get("banner", "default_font") == "slant"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_failed_reset_rolls_back_user_and_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glyph_forge import persistence

    path = tmp_path / "settings.json"
    manager = ConfigManager(user_path=path)
    manager.set("banner", "default_font", "small")
    manager.set("image", "default_width", 72, scope=ConfigScope.RUNTIME)
    monkeypatch.setattr(
        persistence.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(ConfigPersistenceError, match="boom"):
        manager.reset_to_defaults()

    assert manager.get("banner", "default_font") == "small"
    assert manager.get("image", "default_width") == 72


def test_concurrent_updates_always_leave_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    manager = ConfigManager(user_path=path)
    threads = [
        threading.Thread(
            target=manager.set,
            args=("plugin-test", f"value-{index}", index),
        )
        for index in range(12)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted["settings"]["plugin-test"]) == 12


def test_top_level_configuration_names_are_unambiguous() -> None:
    from glyph_forge import get_config, get_profile_config, get_settings

    assert isinstance(get_settings(), ConfigManager)
    assert get_profile_config("minimal")["optimization_level"] == 1
    with pytest.warns(DeprecationWarning, match="get_profile_config"):
        legacy = get_config("minimal")
    assert legacy["optimization_level"] == 1
