"""Versioned, portable project and render-preset documents.

Projects deliberately contain only relative asset references and immutable
render variants.  A :class:`ProjectSession` adds bounded editing history and a
crash-recovery sidecar without weakening the serialized contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, cast

from .config.settings import user_config_directory
from .contracts import RenderContractError, RenderRequest
from .persistence import AtomicWriteError, atomic_write_text

PROJECT_SCHEMA = "glyph-forge-project"
PROJECT_SCHEMA_VERSION = 1
PRESET_SCHEMA = "glyph-forge-preset"
PRESET_SCHEMA_VERSION = 1
RECENTS_SCHEMA = "glyph-forge-recents"
RECENTS_SCHEMA_VERSION = 1
RECOVERY_SCHEMA = "glyph-forge-recovery"
RECOVERY_SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".glyphforge.json"
PRESET_SUFFIX = ".glyphpreset.json"
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_VARIANTS = 256
MAX_HISTORY = 100
MAX_RECENTS = 20

_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?\Z")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class ProjectError(Exception):
    """Base class for public project workflow failures."""


class ProjectValidationError(ProjectError, ValueError):
    """A project or preset violates its versioned contract."""


class ProjectPersistenceError(ProjectError, OSError):
    """A valid project or preset could not be loaded or persisted."""


class ProjectRecoveryError(ProjectError):
    """An autosave sidecar is invalid, stale, or cannot be recovered."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _validated_timestamp(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProjectValidationError(f"{name} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProjectValidationError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProjectValidationError(f"{name} must include a timezone")
    return value


def _bounded_string(value: str, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ProjectValidationError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ProjectValidationError(f"{name} cannot be empty")
    if len(normalized) > maximum:
        raise ProjectValidationError(f"{name} cannot exceed {maximum} characters")
    return normalized


def _json_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 12:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return len(value) <= 1024 and all(
            _json_value(item, depth=depth + 1) for item in value
        )
    if isinstance(value, dict):
        return len(value) <= 1024 and all(
            isinstance(key, str)
            and len(key) <= 256
            and _json_value(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    result = dict(value or {})
    if not _json_value(result):
        raise ProjectValidationError(
            "metadata must contain bounded, finite JSON-compatible values"
        )
    return MappingProxyType(result)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectValidationError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    values: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    name: str,
) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required - optional
    if missing:
        raise ProjectValidationError(f"{name} is missing {', '.join(sorted(missing))}")
    if unknown:
        raise ProjectValidationError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )


def _encode(values: Mapping[str, Any]) -> str:
    try:
        encoded = (
            json.dumps(
                values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ProjectValidationError(
            f"Document is not JSON serializable: {exc}"
        ) from exc
    if len(encoded.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ProjectValidationError(
            f"Document exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB limit"
        )
    return encoded


def _read_json(path: Path, *, kind: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if size > MAX_DOCUMENT_BYTES:
            raise ProjectValidationError(
                f"{kind} exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB limit"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except ProjectValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectPersistenceError(f"Could not load {kind} {path}: {exc}") from exc
    return _mapping(value, kind)


def _write_json(path: Path, values: Mapping[str, Any], *, kind: str) -> Path:
    try:
        return atomic_write_text(path, _encode(values))
    except AtomicWriteError as exc:
        raise ProjectPersistenceError(f"Could not save {kind} {path}: {exc}") from exc


def _document_digest(document: "GlyphProject") -> str:
    return hashlib.sha256(_encode(document.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AssetReference:
    """Portable path to one project asset, always relative to the project."""

    path: str
    kind: str = "image"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ProjectValidationError("asset path cannot be empty")
        if "\x00" in self.path or "\\" in self.path:
            raise ProjectValidationError("asset path must use portable forward slashes")
        normalized_path = unicodedata.normalize("NFC", self.path)
        candidate = PurePosixPath(normalized_path)
        if candidate.is_absolute() or candidate == PurePosixPath("."):
            raise ProjectValidationError("asset path must be relative")
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise ProjectValidationError(
                "asset path cannot contain empty, current, or parent segments"
            )
        if any(":" in part for part in candidate.parts):
            raise ProjectValidationError("asset path cannot contain a drive prefix")
        for part in candidate.parts:
            if (
                part.rstrip(" .") != part
                or Path(part).stem.casefold() in _WINDOWS_RESERVED
                or any(
                    ord(character) < 32 or character in '<>"|?*' for character in part
                )
            ):
                raise ProjectValidationError(
                    f"asset path segment {part!r} is not portable across operating systems"
                )
        normalized_kind = _bounded_string(
            self.kind, "asset kind", maximum=32
        ).casefold()
        if not _IDENTIFIER.fullmatch(normalized_kind):
            raise ProjectValidationError("asset kind must be a portable identifier")
        object.__setattr__(self, "path", candidate.as_posix())
        object.__setattr__(self, "kind", normalized_kind)

    def resolve(self, project_path: str | os.PathLike[str]) -> Path:
        """Resolve this reference while enforcing project-directory containment."""

        root = Path(project_path).expanduser().resolve().parent
        resolved = (root / Path(*PurePosixPath(self.path).parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:  # pragma: no cover - constructor is the first guard
            raise ProjectValidationError(
                "asset path escapes the project directory"
            ) from exc
        return resolved

    @classmethod
    def from_path(
        cls,
        asset_path: str | os.PathLike[str],
        project_path: str | os.PathLike[str],
        *,
        kind: str = "image",
    ) -> "AssetReference":
        """Create a portable reference for an asset already inside the project."""

        root = Path(project_path).expanduser().resolve().parent
        source = Path(asset_path).expanduser().resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ProjectValidationError(
                "asset must be inside the project directory; copy it into an "
                "assets folder before creating a portable project"
            ) from exc
        return cls(PurePosixPath(*relative.parts).as_posix(), kind=kind)

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AssetReference":
        selected = _mapping(values, "asset")
        _exact_keys(
            selected,
            required={"path", "kind"},
            name="asset",
        )
        return cls(path=selected["path"], kind=selected["kind"])


@dataclass(frozen=True, slots=True)
class RenderVariant:
    """Named non-destructive render settings within a project."""

    identifier: str
    name: str
    request: RenderRequest = field(default_factory=RenderRequest)

    def __post_init__(self) -> None:
        identifier = _bounded_string(
            self.identifier, "variant identifier", maximum=64
        ).casefold()
        if not _IDENTIFIER.fullmatch(identifier):
            raise ProjectValidationError(
                "variant identifier must contain lowercase letters, numbers, '.', "
                "'_', or '-'"
            )
        if not isinstance(self.request, RenderRequest):
            raise ProjectValidationError("variant request must be a RenderRequest")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "name", _bounded_string(self.name, "variant name"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "name": self.name,
            "request": self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RenderVariant":
        selected = _mapping(values, "variant")
        _exact_keys(
            selected,
            required={"id", "name", "request"},
            name="variant",
        )
        try:
            request = RenderRequest.from_dict(
                _mapping(selected["request"], "variant request")
            )
        except RenderContractError as exc:
            raise ProjectValidationError(f"Invalid variant request: {exc}") from exc
        return cls(
            identifier=selected["id"],
            name=selected["name"],
            request=request,
        )


@dataclass(frozen=True, slots=True)
class GlyphProject:
    """Immutable version-one creative project document."""

    name: str
    source: AssetReference
    variants: tuple[RenderVariant, ...]
    active_variant: str
    created_at: str
    updated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=PROJECT_SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_SCHEMA_VERSION or isinstance(
            self.schema_version, bool
        ):
            raise ProjectValidationError(
                f"Unsupported project schema version {self.schema_version!r}; "
                f"expected {PROJECT_SCHEMA_VERSION}"
            )
        if not isinstance(self.source, AssetReference):
            raise ProjectValidationError("source must be an AssetReference")
        if not isinstance(self.variants, tuple):
            object.__setattr__(self, "variants", tuple(self.variants))
        if not all(isinstance(variant, RenderVariant) for variant in self.variants):
            raise ProjectValidationError("every variant must be a RenderVariant")
        if not 1 <= len(self.variants) <= MAX_VARIANTS:
            raise ProjectValidationError(
                f"projects require between 1 and {MAX_VARIANTS} variants"
            )
        identifiers = [variant.identifier for variant in self.variants]
        if len(set(identifiers)) != len(identifiers):
            raise ProjectValidationError("variant identifiers must be unique")
        active = _bounded_string(
            self.active_variant, "active variant", maximum=64
        ).casefold()
        if active not in identifiers:
            raise ProjectValidationError("active variant does not exist")
        object.__setattr__(self, "name", _bounded_string(self.name, "project name"))
        object.__setattr__(self, "active_variant", active)
        object.__setattr__(
            self, "created_at", _validated_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "updated_at", _validated_timestamp(self.updated_at, "updated_at")
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def active(self) -> RenderVariant:
        return next(
            variant
            for variant in self.variants
            if variant.identifier == self.active_variant
        )

    @classmethod
    def create(
        cls,
        name: str,
        source: AssetReference,
        request: RenderRequest | None = None,
        *,
        now: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GlyphProject":
        timestamp = now or _utc_now()
        return cls(
            name=name,
            source=source,
            variants=(RenderVariant("default", "Default", request or RenderRequest()),),
            active_variant="default",
            created_at=timestamp,
            updated_at=timestamp,
            metadata=metadata or {},
        )

    def with_updates(self, *, now: str | None = None, **updates: Any) -> "GlyphProject":
        return replace(self, updated_at=now or _utc_now(), **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROJECT_SCHEMA,
            "schema_version": self.schema_version,
            "name": self.name,
            "source": self.source.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "active_variant": self.active_variant,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GlyphProject":
        selected = _mapping(values, "project")
        _exact_keys(
            selected,
            required={
                "schema",
                "schema_version",
                "name",
                "source",
                "variants",
                "active_variant",
                "created_at",
                "updated_at",
                "metadata",
            },
            name="project",
        )
        if selected["schema"] != PROJECT_SCHEMA:
            raise ProjectValidationError(f"Not a {PROJECT_SCHEMA!r} document")
        raw_variants = selected["variants"]
        if not isinstance(raw_variants, list):
            raise ProjectValidationError("variants must be an array")
        return cls(
            name=selected["name"],
            source=AssetReference.from_dict(_mapping(selected["source"], "source")),
            variants=tuple(
                RenderVariant.from_dict(_mapping(item, "variant"))
                for item in raw_variants
            ),
            active_variant=selected["active_variant"],
            created_at=selected["created_at"],
            updated_at=selected["updated_at"],
            metadata=_mapping(selected["metadata"], "metadata"),
            schema_version=selected["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class RenderPreset:
    """Portable named render settings shared by every interface."""

    name: str
    request: RenderRequest = field(default_factory=RenderRequest)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=PRESET_SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        if self.schema_version != PRESET_SCHEMA_VERSION or isinstance(
            self.schema_version, bool
        ):
            raise ProjectValidationError(
                f"Unsupported preset schema version {self.schema_version!r}; "
                f"expected {PRESET_SCHEMA_VERSION}"
            )
        if not isinstance(self.request, RenderRequest):
            raise ProjectValidationError("preset request must be a RenderRequest")
        object.__setattr__(self, "name", _bounded_string(self.name, "preset name"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRESET_SCHEMA,
            "schema_version": self.schema_version,
            "name": self.name,
            "request": self.request.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RenderPreset":
        selected = _mapping(values, "preset")
        _exact_keys(
            selected,
            required={"schema", "schema_version", "name", "request", "metadata"},
            name="preset",
        )
        if selected["schema"] != PRESET_SCHEMA:
            raise ProjectValidationError(f"Not a {PRESET_SCHEMA!r} document")
        try:
            request = RenderRequest.from_dict(
                _mapping(selected["request"], "preset request")
            )
        except RenderContractError as exc:
            raise ProjectValidationError(f"Invalid preset request: {exc}") from exc
        return cls(
            name=selected["name"],
            request=request,
            metadata=_mapping(selected["metadata"], "metadata"),
            schema_version=selected["schema_version"],
        )


def save_project(project: GlyphProject, destination: str | os.PathLike[str]) -> Path:
    """Atomically persist a validated project document."""

    if not isinstance(project, GlyphProject):
        raise ProjectValidationError("project must be a GlyphProject")
    return _write_json(
        Path(destination).expanduser(), project.to_dict(), kind="project"
    )


def load_project(source: str | os.PathLike[str]) -> GlyphProject:
    """Load and validate one project without requiring its asset to exist."""

    path = Path(source).expanduser()
    project = GlyphProject.from_dict(_read_json(path, kind="project"))
    project.source.resolve(path)
    return project


def save_preset(preset: RenderPreset, destination: str | os.PathLike[str]) -> Path:
    """Atomically persist a validated render preset."""

    if not isinstance(preset, RenderPreset):
        raise ProjectValidationError("preset must be a RenderPreset")
    return _write_json(Path(destination).expanduser(), preset.to_dict(), kind="preset")


def load_preset(source: str | os.PathLike[str]) -> RenderPreset:
    """Load and validate one render preset."""

    return RenderPreset.from_dict(_read_json(Path(source).expanduser(), kind="preset"))


def recovery_path(project_path: str | os.PathLike[str]) -> Path:
    path = Path(project_path).expanduser()
    return path.with_name(f".{path.name}.autosave")


class ProjectSession:
    """Thread-safe editing state with history and crash-safe autosave."""

    def __init__(
        self,
        project: GlyphProject,
        path: str | os.PathLike[str],
        *,
        history_limit: int = MAX_HISTORY,
        autosave_delay: float | None = 1.0,
        base_digest: str | None = None,
        recovered: bool = False,
    ) -> None:
        if not 1 <= history_limit <= 10_000:
            raise ProjectValidationError("history_limit must be between 1 and 10000")
        if autosave_delay is not None and (
            isinstance(autosave_delay, bool)
            or not math.isfinite(autosave_delay)
            or autosave_delay < 0
        ):
            raise ProjectValidationError(
                "autosave_delay must be a non-negative finite number or None"
            )
        self._lock = threading.RLock()
        self._project = project
        self.path = Path(path).expanduser()
        self.history_limit = history_limit
        self.autosave_delay = autosave_delay
        self._base_digest = base_digest or _document_digest(project)
        self._undo: list[GlyphProject] = []
        self._redo: list[GlyphProject] = []
        self._timer: threading.Timer | None = None
        self._closed = False
        self._dirty = recovered
        self._last_autosave_error: ProjectError | None = None

    @property
    def project(self) -> GlyphProject:
        with self._lock:
            return self._project

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    @property
    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)

    @property
    def last_autosave_error(self) -> ProjectError | None:
        """Most recent background autosave failure, cleared by a successful save."""

        with self._lock:
            return self._last_autosave_error

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        recover: bool = True,
        history_limit: int = MAX_HISTORY,
        autosave_delay: float | None = 1.0,
    ) -> "ProjectSession":
        selected_path = Path(path).expanduser()
        saved = load_project(selected_path)
        digest = _document_digest(saved)
        project = saved
        recovered = False
        sidecar = recovery_path(selected_path)
        if recover and sidecar.is_file():
            values = _read_json(sidecar, kind="project recovery")
            _exact_keys(
                values,
                required={
                    "schema",
                    "schema_version",
                    "base_sha256",
                    "saved_at",
                    "project",
                },
                name="project recovery",
            )
            if (
                values["schema"] != RECOVERY_SCHEMA
                or values["schema_version"] != RECOVERY_SCHEMA_VERSION
            ):
                raise ProjectRecoveryError("Unsupported project recovery document")
            if values["base_sha256"] != digest:
                raise ProjectRecoveryError(
                    "Autosave is stale because the saved project changed; "
                    "open without recovery or discard the autosave"
                )
            _validated_timestamp(values["saved_at"], "recovery saved_at")
            project = GlyphProject.from_dict(
                _mapping(values["project"], "recovered project")
            )
            project.source.resolve(selected_path)
            recovered = True
        return cls(
            project,
            selected_path,
            history_limit=history_limit,
            autosave_delay=autosave_delay,
            base_digest=digest,
            recovered=recovered,
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise ProjectPersistenceError("project session is closed")

    def _schedule_autosave(self) -> None:
        if self.autosave_delay is None:
            return
        if self._timer is not None:
            self._timer.cancel()
        if self.autosave_delay == 0:
            self.checkpoint()
            return
        timer = threading.Timer(self.autosave_delay, self._checkpoint_from_timer)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _checkpoint_from_timer(self) -> None:
        try:
            self.checkpoint()
        except ProjectError as exc:
            # Explicit flush/save calls surface persistence failures.  Background
            # autosave cannot safely raise into the UI thread, so retain it for
            # status surfaces and diagnostics.
            with self._lock:
                self._last_autosave_error = exc

    def _replace_project(self, project: GlyphProject, *, record: bool) -> GlyphProject:
        self._assert_open()
        if project == self._project:
            return self._project
        if record:
            self._undo.append(self._project)
            del self._undo[: -self.history_limit]
            self._redo.clear()
        self._project = project
        self._dirty = True
        self._schedule_autosave()
        return project

    def update_active_request(self, request: RenderRequest) -> GlyphProject:
        """Replace the active variant settings as one undoable operation."""

        if not isinstance(request, RenderRequest):
            raise ProjectValidationError("request must be a RenderRequest")
        with self._lock:
            if request == self._project.active.request:
                return self._project
            variants = tuple(
                replace(variant, request=request)
                if variant.identifier == self._project.active_variant
                else variant
                for variant in self._project.variants
            )
            return self._replace_project(
                self._project.with_updates(variants=variants), record=True
            )

    def add_variant(
        self,
        identifier: str,
        name: str,
        request: RenderRequest | None = None,
        *,
        activate: bool = True,
    ) -> GlyphProject:
        """Add a variant, copying the active request when none is supplied."""

        with self._lock:
            variant = RenderVariant(
                identifier,
                name,
                request or self._project.active.request,
            )
            if any(
                item.identifier == variant.identifier for item in self._project.variants
            ):
                raise ProjectValidationError(
                    f"variant {variant.identifier!r} already exists"
                )
            if len(self._project.variants) >= MAX_VARIANTS:
                raise ProjectValidationError(
                    f"projects cannot exceed {MAX_VARIANTS} variants"
                )
            return self._replace_project(
                self._project.with_updates(
                    variants=(*self._project.variants, variant),
                    active_variant=(
                        variant.identifier if activate else self._project.active_variant
                    ),
                ),
                record=True,
            )

    def remove_variant(self, identifier: str) -> GlyphProject:
        """Remove one variant while retaining at least one render state."""

        selected = identifier.casefold()
        with self._lock:
            if len(self._project.variants) == 1:
                raise ProjectValidationError(
                    "the last project variant cannot be removed"
                )
            variants = tuple(
                item for item in self._project.variants if item.identifier != selected
            )
            if len(variants) == len(self._project.variants):
                raise ProjectValidationError(f"unknown variant {identifier!r}")
            active = self._project.active_variant
            if active == selected:
                active = variants[0].identifier
            return self._replace_project(
                self._project.with_updates(
                    variants=variants,
                    active_variant=active,
                ),
                record=True,
            )

    def select_variant(self, identifier: str) -> GlyphProject:
        """Select one existing non-destructive variant."""

        selected = identifier.casefold()
        with self._lock:
            if not any(item.identifier == selected for item in self._project.variants):
                raise ProjectValidationError(f"unknown variant {identifier!r}")
            if selected == self._project.active_variant:
                return self._project
            return self._replace_project(
                self._project.with_updates(active_variant=selected), record=True
            )

    def undo(self) -> GlyphProject:
        with self._lock:
            self._assert_open()
            if not self._undo:
                raise ProjectValidationError("nothing to undo")
            previous = self._undo.pop()
            self._redo.append(self._project)
            self._project = previous
            self._dirty = True
            self._schedule_autosave()
            return previous

    def redo(self) -> GlyphProject:
        with self._lock:
            self._assert_open()
            if not self._redo:
                raise ProjectValidationError("nothing to redo")
            following = self._redo.pop()
            self._undo.append(self._project)
            self._project = following
            self._dirty = True
            self._schedule_autosave()
            return following

    def checkpoint(self) -> Path:
        """Synchronously write the current state to its recovery sidecar."""

        with self._lock:
            self._assert_open()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            values = {
                "schema": RECOVERY_SCHEMA,
                "schema_version": RECOVERY_SCHEMA_VERSION,
                "base_sha256": self._base_digest,
                "saved_at": _utc_now(),
                "project": self._project.to_dict(),
            }
            result = _write_json(
                recovery_path(self.path), values, kind="project recovery"
            )
            self._last_autosave_error = None
            return result

    def save(self) -> Path:
        """Persist the project, clear recovery, and establish a new base digest."""

        with self._lock:
            self._assert_open()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            result = save_project(self._project, self.path)
            self._base_digest = _document_digest(self._project)
            self._dirty = False
            self._last_autosave_error = None
            try:
                recovery_path(self.path).unlink(missing_ok=True)
            except OSError as exc:
                raise ProjectPersistenceError(
                    f"Project saved, but recovery could not be cleared: {exc}"
                ) from exc
            return result

    def discard_recovery(self) -> None:
        try:
            recovery_path(self.path).unlink(missing_ok=True)
        except OSError as exc:
            raise ProjectPersistenceError(f"Could not discard recovery: {exc}") from exc

    def close(self, *, checkpoint: bool = True) -> None:
        """Stop pending timers, optionally preserving unsaved edits for recovery."""

        with self._lock:
            if self._closed:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if checkpoint and self._dirty:
                self.checkpoint()
            self._closed = True

    def __enter__(self) -> "ProjectSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RecentProject:
    path: Path
    accessed_at: str


class RecentProjectStore:
    """Bounded, platform-native recent-project history."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        limit: int = MAX_RECENTS,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if not 1 <= limit <= 1000:
            raise ProjectValidationError("recent project limit must be 1–1000")
        self.path = (
            Path(path).expanduser()
            if path is not None
            else user_config_directory() / "recent_projects.json"
        )
        self.limit = limit
        self.clock = clock
        self._lock = threading.RLock()

    def list(self, *, existing_only: bool = False) -> tuple[RecentProject, ...]:
        with self._lock:
            if not self.path.is_file():
                return ()
            values = _read_json(self.path, kind="recent projects")
            _exact_keys(
                values,
                required={"schema", "schema_version", "projects"},
                name="recent projects",
            )
            if (
                values["schema"] != RECENTS_SCHEMA
                or values["schema_version"] != RECENTS_SCHEMA_VERSION
            ):
                raise ProjectValidationError("Unsupported recent-project store version")
            raw_projects = values["projects"]
            if not isinstance(raw_projects, list) or len(raw_projects) > 1000:
                raise ProjectValidationError("recent projects must be a bounded array")
            result: list[RecentProject] = []
            seen: set[Path] = set()
            for raw in raw_projects:
                item = _mapping(raw, "recent project")
                _exact_keys(
                    item,
                    required={"path", "accessed_at"},
                    name="recent project",
                )
                if not isinstance(item["path"], str) or not item["path"]:
                    raise ProjectValidationError("recent project path must be a string")
                if "\x00" in item["path"]:
                    raise ProjectValidationError(
                        "recent project path contains a null byte"
                    )
                path = Path(item["path"]).expanduser().resolve()
                accessed_at = _validated_timestamp(
                    item["accessed_at"], "recent project accessed_at"
                )
                if path in seen or (existing_only and not path.is_file()):
                    continue
                seen.add(path)
                result.append(RecentProject(path, accessed_at))
            return tuple(result[: self.limit])

    def _save(self, projects: Sequence[RecentProject]) -> Path:
        values = {
            "schema": RECENTS_SCHEMA,
            "schema_version": RECENTS_SCHEMA_VERSION,
            "projects": [
                {"path": str(item.path), "accessed_at": item.accessed_at}
                for item in projects[: self.limit]
            ],
        }
        return _write_json(self.path, values, kind="recent projects")

    def touch(self, project_path: str | os.PathLike[str]) -> Path:
        selected = Path(project_path).expanduser().resolve()
        with self._lock:
            projects = [item for item in self.list() if item.path != selected]
            projects.insert(0, RecentProject(selected, self.clock()))
            return self._save(projects)

    def remove(self, project_path: str | os.PathLike[str]) -> Path:
        selected = Path(project_path).expanduser().resolve()
        with self._lock:
            return self._save([item for item in self.list() if item.path != selected])

    def prune(self) -> Path:
        with self._lock:
            return self._save(self.list(existing_only=True))


__all__ = [
    "AssetReference",
    "GlyphProject",
    "MAX_HISTORY",
    "MAX_RECENTS",
    "MAX_VARIANTS",
    "PRESET_SCHEMA",
    "PRESET_SCHEMA_VERSION",
    "PROJECT_SCHEMA",
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_SUFFIX",
    "PRESET_SUFFIX",
    "ProjectError",
    "ProjectPersistenceError",
    "ProjectRecoveryError",
    "ProjectSession",
    "ProjectValidationError",
    "RecentProject",
    "RecentProjectStore",
    "RenderPreset",
    "RenderVariant",
    "load_preset",
    "load_project",
    "recovery_path",
    "save_preset",
    "save_project",
]
