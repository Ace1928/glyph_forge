"""Thread-safe layered configuration for Glyph Forge.

Configuration is resolved in this order: built-in defaults, optional system
configuration, user configuration, and in-memory runtime overrides. Only user
overrides are persisted. Reads never create directories or files.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeAlias

from glyph_forge.exceptions import GlyphForgeConfigError, GlyphForgeInputError

logger = logging.getLogger(__name__)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ConfigStore: TypeAlias = dict[str, dict[str, JSONValue]]


class ConfigScope(Enum):
    """Writable configuration lifetimes."""

    SYSTEM = "system"
    USER = "user"
    RUNTIME = "runtime"


class ConfigManager:
    """Resolve and persist Glyph Forge settings with deterministic precedence.

    ``SYSTEM`` is accepted as an enum value for compatibility and inspection,
    but callers cannot write it. Administrators manage the system JSON file
    outside the process. ``USER`` writes are persisted atomically. ``RUNTIME``
    writes live only for the lifetime of this object.
    """

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
            "brightness": 1.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "dithering": False,
            "dither_algorithm": "floyd-steinberg",
            "aspect_ratio": 0.55,
            "resample": "lanczos",
            "autocontrast": False,
            "equalize": False,
            "invert_image": False,
            "edge_enhance": False,
            "sharpen": False,
            "blur_radius": 0.0,
            "posterize_bits": None,
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
        """Load available layers without creating filesystem state."""

        self._lock = threading.RLock()
        self._config_paths: dict[ConfigScope, Path | None] = {
            ConfigScope.SYSTEM: system_path if system_path is not None else self._system_path(),
            ConfigScope.USER: user_path if user_path is not None else self._user_path(),
            ConfigScope.RUNTIME: None,
        }
        self._layers: dict[ConfigScope, ConfigStore] = {
            ConfigScope.SYSTEM: self._load_file(self._config_paths[ConfigScope.SYSTEM]),
            ConfigScope.USER: self._load_file(self._config_paths[ConfigScope.USER]),
            ConfigScope.RUNTIME: {},
        }
        self.config: ConfigStore = {}
        self._rebuild()

    @staticmethod
    def _system_path() -> Path | None:
        explicit = os.environ.get("GLYPH_FORGE_SYSTEM_CONFIG")
        if explicit:
            return Path(explicit).expanduser()
        if os.name == "nt":
            root = os.environ.get("PROGRAMDATA")
            return Path(root) / "Glyph Forge" / "config.json" if root else None
        return Path("/etc/glyph-forge/config.json")

    @staticmethod
    def _user_path() -> Path:
        explicit = os.environ.get("GLYPH_FORGE_CONFIG_FILE")
        if explicit:
            return Path(explicit).expanduser()
        if os.name == "nt":
            root = os.environ.get("APPDATA")
            base = Path(root) if root else Path.home() / "AppData" / "Roaming"
            return base / "Glyph Forge" / "config.json"
        root = os.environ.get("XDG_CONFIG_HOME")
        base = Path(root).expanduser() if root else Path.home() / ".config"
        return base / "glyph-forge" / "config.json"

    @staticmethod
    def _validate_store(value: Any, *, source: Path) -> ConfigStore:
        if not isinstance(value, dict):
            raise GlyphForgeConfigError(f"configuration root in {source} must be an object")
        result: ConfigStore = {}
        for section, settings in value.items():
            if not isinstance(section, str) or not isinstance(settings, dict):
                raise GlyphForgeConfigError(
                    f"configuration sections in {source} must be named objects"
                )
            if not all(isinstance(key, str) for key in settings):
                raise GlyphForgeConfigError(f"configuration keys in {source} must be strings")
            result[section] = copy.deepcopy(settings)
        return result

    def _load_file(self, path: Path | None) -> ConfigStore:
        if path is None or not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return self._validate_store(json.load(handle), source=path)
        except (GlyphForgeConfigError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring invalid configuration %s: %s", path, exc)
            return {}

    @staticmethod
    def _merge(base: ConfigStore, overlay: ConfigStore) -> None:
        for section, values in overlay.items():
            base.setdefault(section, {}).update(copy.deepcopy(values))

    def _rebuild(self) -> None:
        merged = copy.deepcopy(self.DEFAULT_CONFIG)
        self._merge(merged, self._layers[ConfigScope.SYSTEM])
        self._merge(merged, self._layers[ConfigScope.USER])
        self._merge(merged, self._layers[ConfigScope.RUNTIME])
        self.config = merged

    @staticmethod
    def _validate_name(value: str, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise GlyphForgeInputError(f"configuration {label} must be a non-empty string")

    @staticmethod
    def _validate_value(value: JSONValue) -> None:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise GlyphForgeInputError("configuration values must be finite JSON data") from exc

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Return a defensive copy of a resolved setting or ``default``."""

        with self._lock:
            return copy.deepcopy(self.config.get(section, {}).get(key, default))

    def set(
        self,
        section: str,
        key: str,
        value: JSONValue,
        scope: ConfigScope = ConfigScope.USER,
    ) -> None:
        """Set a user or runtime override and persist user changes atomically."""

        self._validate_name(section, "section")
        self._validate_name(key, "key")
        self._validate_value(value)
        if scope is ConfigScope.SYSTEM:
            raise GlyphForgeConfigError("system configuration is read-only")
        if scope not in {ConfigScope.USER, ConfigScope.RUNTIME}:
            raise GlyphForgeInputError(f"unsupported configuration scope: {scope!r}")

        with self._lock:
            layer = self._layers[scope]
            layer.setdefault(section, {})[key] = copy.deepcopy(value)
            self._rebuild()
            if scope is ConfigScope.USER:
                self._save_user_layer()

    def _save_user_layer(self) -> None:
        path = self._config_paths[ConfigScope.USER]
        if path is None:
            raise GlyphForgeConfigError("no user configuration path is available")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    self._layers[ConfigScope.USER],
                    temporary,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            if os.name != "nt":
                path.chmod(0o600)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise GlyphForgeConfigError(
                f"could not save user configuration to {path}: {exc}"
            ) from exc

    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """Remove user and runtime overrides globally or for one section."""

        if section is not None:
            self._validate_name(section, "section")
        with self._lock:
            for scope in (ConfigScope.USER, ConfigScope.RUNTIME):
                if section is None:
                    self._layers[scope].clear()
                else:
                    self._layers[scope].pop(section, None)
            self._rebuild()
            self._save_user_layer()

    def get_sections(self) -> list[str]:
        """Return resolved section names in deterministic order."""

        with self._lock:
            return sorted(self.config)

    def get_section(self, section: str) -> dict[str, JSONValue]:
        """Return a defensive copy of a resolved section."""

        with self._lock:
            return copy.deepcopy(self.config.get(section, {}))


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


__all__ = ["ConfigManager", "ConfigScope", "get_config"]
