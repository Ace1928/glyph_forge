"""Versioned, platform-aware, atomic user configuration."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

from ..persistence import AtomicWriteError, atomic_write_bytes
from ..visual_defaults import DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST

logger = logging.getLogger(__name__)

CONFIG_SCHEMA_VERSION = 1
ConfigValue = Union[None, str, int, float, bool, List[str], Dict[str, Any]]
ConfigStore = Dict[str, Dict[str, ConfigValue]]


class ConfigError(Exception):
    """Base class for configuration failures."""


class ConfigValidationError(ConfigError, ValueError):
    """A configuration value does not satisfy its declared contract."""


class ConfigPersistenceError(ConfigError, OSError):
    """A valid configuration update could not be persisted safely."""


class ConfigScope(str, Enum):
    """Storage lifetime for a setting update."""

    SYSTEM = "system"
    USER = "user"
    RUNTIME = "runtime"


def user_config_directory() -> Path:
    """Return the native per-user configuration directory without creating it."""

    explicit = os.environ.get("GLYPH_FORGE_CONFIG_HOME")
    if explicit:
        return Path(explicit).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return root / "GlyphForge"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Glyph Forge"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "glyph_forge"


def default_user_config_path() -> Path:
    """Return the canonical versioned settings file location."""

    explicit = os.environ.get("GLYPH_FORGE_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser()
    return user_config_directory() / "user_config.json"


def _legacy_user_config_path() -> Path | None:
    """Return the pre-0.4 user settings path when it differs from canonical."""

    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return root / "GLYPH_Forge" / "user_config.json"
    if sys.platform == "darwin":
        xdg = os.environ.get("XDG_CONFIG_HOME")
        root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
        return root / "glyph_forge" / "user_config.json"
    return None


def _system_config_path() -> Path:
    system = Path("/etc/glyph_forge/system_config.json")
    if system.is_file():
        return system
    return Path(__file__).with_name("system_config.json")


def _json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_compatible(item)
            for key, item in value.items()
        )
    return False


_INTEGER_RANGES: dict[tuple[str, str], tuple[int, int]] = {
    ("banner", "default_width"): (1, 4096),
    ("banner", "cache_size"): (0, 100_000),
    ("banner", "cache_ttl"): (0, 31_536_000),
    ("image", "default_width"): (1, 4096),
    ("image", "max_width"): (1, 4096),
    ("image", "max_threads"): (1, 256),
    ("performance", "optimization_level"): (1, 5),
}
_FLOAT_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("image", "brightness"): (0.0, 2.0),
    ("image", "contrast"): (0.0, 2.0),
}
_BOOLEAN_FIELDS = {
    ("banner", "cache_enabled"),
    ("banner", "unicode_enabled"),
    ("image", "dithering"),
    ("image", "parallel_processing"),
    ("io", "auto_detect_terminal"),
    ("io", "color_output"),
    ("io", "backup_files"),
    ("performance", "cache_enabled"),
    ("performance", "lazy_loading"),
    ("performance", "debug_mode"),
}
_STRING_FIELDS = {
    ("banner", "default_font"),
    ("banner", "default_style"),
    ("image", "default_charset"),
    ("io", "output_format"),
    ("io", "temp_directory"),
}


def _validated_integer(
    section: str,
    key: str,
    value: Any,
    bounds: tuple[int, int],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{section}.{key} must be an integer")
    minimum, maximum = bounds
    if not minimum <= value <= maximum:
        raise ConfigValidationError(
            f"{section}.{key} must be between {minimum} and {maximum}"
        )
    return value


def _validated_float(
    section: str,
    key: str,
    value: Any,
    bounds: tuple[float, float],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{section}.{key} must be a number")
    minimum, maximum = bounds
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ConfigValidationError(
            f"{section}.{key} must be between {minimum} and {maximum}"
        )
    return numeric


def _validated_known_field(
    section: str,
    key: str,
    value: Any,
) -> Any:
    field = (section, key)
    if field in _INTEGER_RANGES:
        return _validated_integer(section, key, value, _INTEGER_RANGES[field])
    if field in _FLOAT_RANGES:
        return _validated_float(section, key, value, _FLOAT_RANGES[field])
    if field in _BOOLEAN_FIELDS and not isinstance(value, bool):
        raise ConfigValidationError(f"{section}.{key} must be true or false")
    if field in _STRING_FIELDS and not isinstance(value, str):
        raise ConfigValidationError(f"{section}.{key} must be a string")
    return value


def _validate_value(section: str, key: str, value: Any) -> ConfigValue:
    if not section or not key:
        raise ConfigValidationError("Configuration section and key cannot be empty")
    value = _validated_known_field(section, key, value)
    if not _json_compatible(value):
        raise ConfigValidationError(
            f"{section}.{key} must contain only finite JSON-compatible values"
        )
    return cast(ConfigValue, value)


class ConfigManager:
    """Thread-safe layered settings with schema migration and atomic writes."""

    DEFAULT_CONFIG: ConfigStore = {
        "banner": {
            "default_font": "slant",
            "default_width": 80,
            "default_style": "minimal",
            "cache_enabled": True,
            "cache_size": 100,
            "cache_ttl": 3600,
            "unicode_enabled": True,
        },
        "image": {
            "default_charset": "general",
            "default_width": 100,
            "max_width": 500,
            "brightness": DEFAULT_BRIGHTNESS,
            "contrast": DEFAULT_CONTRAST,
            "dithering": False,
            "parallel_processing": True,
            "max_threads": 4,
        },
        "io": {
            "output_format": "text",
            "auto_detect_terminal": True,
            "color_output": True,
            "backup_files": True,
            "temp_directory": "",
        },
        "performance": {
            "optimization_level": 3,
            "cache_enabled": True,
            "lazy_loading": True,
            "debug_mode": False,
        },
    }

    def __init__(
        self,
        *,
        user_path: Path | None = None,
        system_path: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._legacy_discovery_enabled = user_path is None and not any(
            os.environ.get(name)
            for name in ("GLYPH_FORGE_CONFIG_FILE", "GLYPH_FORGE_CONFIG_HOME")
        )
        self._config_paths: dict[ConfigScope, Path | None] = {
            ConfigScope.SYSTEM: system_path or _system_config_path(),
            ConfigScope.USER: user_path or default_user_config_path(),
            ConfigScope.RUNTIME: None,
        }
        self._system_config = self._read_config(
            self._config_paths[ConfigScope.SYSTEM],
            label="system",
        )
        self._user_config = self._read_config(
            self._user_read_path(),
            label="user",
        )
        self._runtime_config: ConfigStore = {}
        self._dirty_scopes: Set[ConfigScope] = set()
        self.config: ConfigStore = {}
        self._rebuild()

    @property
    def user_path(self) -> Path:
        path = self._config_paths[ConfigScope.USER]
        assert path is not None
        return path

    def _user_read_path(self) -> Path:
        canonical = self.user_path
        if canonical.is_file() or not self._legacy_discovery_enabled:
            return canonical
        legacy = _legacy_user_config_path()
        if legacy is not None and legacy != canonical and legacy.is_file():
            logger.info(
                "Reading legacy user config %s; the next update will migrate it to %s",
                legacy,
                canonical,
            )
            return legacy
        return canonical

    def _read_config(self, path: Path | None, *, label: str) -> ConfigStore:
        if path is None or not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                raise ConfigValidationError("root must be an object")
            if "schema_version" in payload:
                version = payload.get("schema_version")
                if version != CONFIG_SCHEMA_VERSION:
                    raise ConfigValidationError(
                        f"unsupported schema version {version!r}"
                    )
                values = payload.get("settings", {})
            else:
                # Version 0 files stored sections directly. Reading them is the
                # migration; the next user write persists the versioned shape.
                values = payload
            if not isinstance(values, dict):
                raise ConfigValidationError("settings must be an object")
            return self._validated_store(values, label=label)
        except (OSError, json.JSONDecodeError, ConfigValidationError) as exc:
            logger.warning("Ignoring invalid %s config %s: %s", label, path, exc)
            return {}

    def _validated_store(self, values: dict[str, Any], *, label: str) -> ConfigStore:
        result: ConfigStore = {}
        for section, entries in values.items():
            if not isinstance(section, str) or not isinstance(entries, dict):
                logger.warning("Ignoring invalid %s config section %r", label, section)
                continue
            for key, value in entries.items():
                if not isinstance(key, str):
                    logger.warning("Ignoring non-string key in %s.%s", label, section)
                    continue
                try:
                    selected = _validate_value(section, key, value)
                except ConfigValidationError as exc:
                    logger.warning("Ignoring invalid %s setting: %s", label, exc)
                    continue
                result.setdefault(section, {})[key] = copy.deepcopy(selected)
        return result

    @staticmethod
    def _merge(base: ConfigStore, overlay: ConfigStore) -> None:
        for section, values in overlay.items():
            base.setdefault(section, {}).update(copy.deepcopy(values))

    def _rebuild(self) -> None:
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._merge(self.config, self._system_config)
        self._merge(self.config, self._user_config)
        self._merge(self.config, self._runtime_config)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return copy.deepcopy(self.config.get(section, {}).get(key, default))

    def set(
        self,
        section: str,
        key: str,
        value: Any,
        scope: ConfigScope = ConfigScope.USER,
    ) -> None:
        selected_scope = ConfigScope(scope)
        if selected_scope is ConfigScope.SYSTEM:
            raise ConfigValidationError("System configuration is read-only")
        selected = _validate_value(section, key, value)
        with self._lock:
            layer = (
                self._runtime_config
                if selected_scope is ConfigScope.RUNTIME
                else self._user_config
            )
            if selected_scope is ConfigScope.RUNTIME:
                layer.setdefault(section, {})[key] = copy.deepcopy(selected)
                self._rebuild()
                return
            previous_user = copy.deepcopy(self._user_config)
            previous_dirty = set(self._dirty_scopes)
            layer.setdefault(section, {})[key] = copy.deepcopy(selected)
            self._rebuild()
            self._dirty_scopes.add(ConfigScope.USER)
            try:
                self._save_user_config()
            except ConfigPersistenceError:
                self._user_config = previous_user
                self._dirty_scopes = previous_dirty
                self._rebuild()
                raise

    def _save_user_config(self) -> None:
        destination = self.user_path
        document = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "settings": self._user_config,
        }
        encoded = (
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        try:
            atomic_write_bytes(destination, encoded, permissions=0o600)
            self._dirty_scopes.discard(ConfigScope.USER)
        except AtomicWriteError as exc:
            raise ConfigPersistenceError(
                f"Could not save configuration to {destination}: {exc}"
            ) from exc

    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        with self._lock:
            previous_user = copy.deepcopy(self._user_config)
            previous_runtime = copy.deepcopy(self._runtime_config)
            previous_dirty = set(self._dirty_scopes)
            if section is None:
                self._user_config.clear()
                self._runtime_config.clear()
            else:
                self._user_config.pop(section, None)
                self._runtime_config.pop(section, None)
            self._rebuild()
            self._dirty_scopes.add(ConfigScope.USER)
            try:
                self._save_user_config()
            except ConfigPersistenceError:
                self._user_config = previous_user
                self._runtime_config = previous_runtime
                self._dirty_scopes = previous_dirty
                self._rebuild()
                raise

    def reload(self) -> None:
        """Reload persistent layers while retaining session-only overrides."""

        with self._lock:
            self._system_config = self._read_config(
                self._config_paths[ConfigScope.SYSTEM],
                label="system",
            )
            self._user_config = self._read_config(
                self._user_read_path(),
                label="user",
            )
            self._rebuild()

    def get_sections(self) -> List[str]:
        with self._lock:
            return list(self.config)

    def get_section(self, section: str) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.config.get(section, {}))

    def snapshot(self) -> ConfigStore:
        """Return an isolated view suitable for diagnostics or export."""

        with self._lock:
            return copy.deepcopy(self.config)


_config_instance: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config() -> ConfigManager:
    """Return the process-wide configuration manager."""

    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ConfigManager()
    return _config_instance


get_settings = get_config


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ConfigError",
    "ConfigManager",
    "ConfigPersistenceError",
    "ConfigScope",
    "ConfigStore",
    "ConfigValidationError",
    "ConfigValue",
    "default_user_config_path",
    "get_config",
    "get_settings",
    "user_config_directory",
]
