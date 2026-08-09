"""Lazy, failure-isolated discovery for Glyph Forge plugins."""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

from .contracts import (
    PLUGIN_API_VERSION,
    PLUGIN_ENTRY_POINT_GROUP,
    ExportReceipt,
    ExportRequest,
    FrameSourceExtension,
    PluginManifest,
    RendererExtension,
    RendererRequest,
    RenderOutput,
    SourceRequest,
    TransformRequest,
)

PluginKind = Literal["source", "renderer", "transform", "exporter"]
_KIND_ATTRIBUTES: dict[PluginKind, str] = {
    "source": "sources",
    "renderer": "renderers",
    "transform": "transforms",
    "exporter": "exporters",
}
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


class PluginError(RuntimeError):
    """Base class for actionable extension failures."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin or component reference cannot be resolved."""


class PluginConflictError(PluginError):
    """Raised when two installed entry points claim one identifier."""


class PluginLoadError(PluginError):
    """Raised when an explicitly selected plugin cannot be imported."""


class PluginCompatibilityError(PluginError):
    """Raised when a plugin targets a different contract version."""


class PluginContractError(PluginError):
    """Raised when a manifest or component violates its declared contract."""


class PluginExecutionError(PluginError):
    """Raised when extension code fails during an explicit invocation."""


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Serializable plugin status used by diagnostics and user interfaces."""

    identifier: str
    name: str
    version: str
    description: str
    distribution: str | None
    entry_point: str | None
    state: str
    api_version: int | None = None
    sources: tuple[str, ...] = ()
    renderers: tuple[str, ...] = ()
    transforms: tuple[str, ...] = ()
    exporters: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComponentReference:
    """A parsed ``plugin-id/component`` reference and optional resource."""

    plugin: str
    component: str
    resource: str = ""

    @property
    def qualified(self) -> str:
        return f"{self.plugin}/{self.component}"


def _normalize_identifier(value: str, *, label: str) -> str:
    normalized = value.strip().casefold()
    if not _IDENTIFIER.fullmatch(normalized):
        raise PluginContractError(
            f"Invalid {label} {value!r}; use letters, numbers, dot, dash, or underscore"
        )
    return normalized


def parse_component_reference(
    value: str,
    *,
    allow_resource: bool = False,
) -> ComponentReference:
    """Parse an extension reference without importing extension code."""

    reference = value[7:] if value.casefold().startswith("plugin:") else value
    head, separator, resource = reference.partition(":")
    if separator and not allow_resource:
        raise PluginContractError(
            f"Component reference {value!r} cannot contain a resource suffix"
        )
    plugin, slash, component = head.partition("/")
    if not slash or not plugin or not component or "/" in component:
        raise PluginContractError(
            "Plugin components must look like plugin:plugin-id/component"
        )
    return ComponentReference(
        plugin=_normalize_identifier(plugin, label="plugin identifier"),
        component=_normalize_identifier(component, label="component name"),
        resource=resource if separator else "",
    )


def is_plugin_reference(value: object) -> bool:
    return isinstance(value, str) and value.casefold().startswith("plugin:")


def plugins_enabled() -> bool:
    """Return whether automatic third-party entry-point discovery is enabled."""

    value = os.environ.get("GLYPH_FORGE_DISABLE_PLUGINS", "")
    return value.casefold() not in {"1", "true", "yes", "on"}


def _installed_entry_points() -> Iterable[Any]:
    points = metadata.entry_points()
    if hasattr(points, "select"):
        return points.select(group=PLUGIN_ENTRY_POINT_GROUP)
    return cast(Iterable[Any], cast(Any, points).get(PLUGIN_ENTRY_POINT_GROUP, ()))


def _entry_point_distribution(point: Any) -> tuple[str | None, str]:
    distribution = getattr(point, "dist", None)
    if distribution is None:
        return None, ""
    name = distribution.metadata.get("Name") or getattr(distribution, "name", None)
    return (str(name) if name else None), str(getattr(distribution, "version", ""))


class PluginRegistry:
    """Discover metadata cheaply and load only explicitly used plugins.

    Each plugin is imported independently and failures are cached per plugin.
    A broken optional extension therefore cannot prevent core startup or another
    extension from loading.
    """

    def __init__(
        self,
        *,
        discoverer: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self._discoverer = discoverer or _installed_entry_points
        self._lock = threading.RLock()
        self._discovered = False
        self._points: dict[str, Any] = {}
        self._conflicts: dict[str, tuple[Any, ...]] = {}
        self._manifests: dict[str, PluginManifest] = {}
        self._errors: dict[str, PluginError] = {}

    def _discover(self) -> None:
        with self._lock:
            if self._discovered:
                return
            self._discovered = True
            if not plugins_enabled():
                return
            grouped: dict[str, list[Any]] = defaultdict(list)
            try:
                points = tuple(self._discoverer())
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:
                self._errors["<discovery>"] = PluginLoadError(
                    f"Plugin metadata discovery failed: {exc}"
                )
                return
            for point in points:
                try:
                    identifier = _normalize_identifier(
                        str(point.name), label="entry-point name"
                    )
                except PluginContractError as exc:
                    self._errors[f"<invalid:{point.name}>"] = exc
                    continue
                grouped[identifier].append(point)
            for identifier, candidates in grouped.items():
                if len(candidates) == 1:
                    self._points[identifier] = candidates[0]
                else:
                    self._conflicts[identifier] = tuple(candidates)

    @staticmethod
    def _validate_manifest(identifier: str, value: object) -> PluginManifest:
        if not isinstance(value, PluginManifest):
            raise PluginContractError(
                f"Plugin {identifier!r} must export PluginManifest, got "
                f"{type(value).__name__}"
            )
        if (
            not isinstance(value.api_version, int)
            or isinstance(value.api_version, bool)
            or value.api_version != PLUGIN_API_VERSION
        ):
            raise PluginCompatibilityError(
                f"Plugin {identifier!r} uses API {value.api_version}; "
                f"this Glyph Forge supports API {PLUGIN_API_VERSION}"
            )
        if not isinstance(value.name, str) or not value.name.strip():
            raise PluginContractError(f"Plugin {identifier!r} has an empty name")
        if not isinstance(value.version, str) or not value.version.strip():
            raise PluginContractError(f"Plugin {identifier!r} has an empty version")
        if not isinstance(value.description, str):
            raise PluginContractError(
                f"Plugin {identifier!r} description must be a string"
            )
        for kind, attribute in _KIND_ATTRIBUTES.items():
            components = getattr(value, attribute)
            for name, component in components.items():
                if not isinstance(name, str):
                    raise PluginContractError(
                        f"Plugin {identifier!r} {kind} component names must be strings"
                    )
                normalized = _normalize_identifier(name, label=f"{kind} component")
                if name != normalized:
                    raise PluginContractError(
                        f"Plugin {identifier!r} {kind} component {name!r} must use "
                        f"its normalized lowercase spelling {normalized!r}"
                    )
                if not callable(component):
                    raise PluginContractError(
                        f"Plugin {identifier!r} {kind} {name!r} is not callable"
                    )
        return value

    def register(
        self,
        identifier: str,
        manifest: PluginManifest,
        *,
        replace: bool = False,
    ) -> None:
        """Register an in-process plugin for embedding, tests, or notebooks."""

        normalized = _normalize_identifier(identifier, label="plugin identifier")
        validated = self._validate_manifest(normalized, manifest)
        with self._lock:
            self._discover()
            occupied = (
                normalized in self._manifests
                or normalized in self._points
                or normalized in self._conflicts
            )
            if occupied and not replace:
                raise PluginConflictError(
                    f"Plugin identifier {normalized!r} is already registered"
                )
            self._points.pop(normalized, None)
            self._conflicts.pop(normalized, None)
            self._errors.pop(normalized, None)
            self._manifests[normalized] = validated

    def load(self, identifier: str) -> PluginManifest:
        """Load and validate one selected plugin, caching its result or failure."""

        normalized = _normalize_identifier(identifier, label="plugin identifier")
        with self._lock:
            self._discover()
            if normalized in self._manifests:
                return self._manifests[normalized]
            if not plugins_enabled():
                raise PluginNotFoundError(
                    "External plugins are disabled by GLYPH_FORGE_DISABLE_PLUGINS"
                )
            if normalized in self._errors:
                raise self._errors[normalized]
            if normalized in self._conflicts:
                conflict_error = PluginConflictError(
                    f"Multiple installed plugins use identifier {normalized!r}"
                )
                self._errors[normalized] = conflict_error
                raise conflict_error
            point = self._points.get(normalized)
            if point is None:
                discovery_error = self._errors.get("<discovery>")
                if discovery_error is not None:
                    raise discovery_error
                raise PluginNotFoundError(f"Plugin {normalized!r} is not installed")
            try:
                exported = point.load()
                candidate = exported() if callable(exported) else exported
                manifest = self._validate_manifest(normalized, candidate)
            except PluginError as exc:
                self._errors[normalized] = exc
                raise
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:
                load_error = PluginLoadError(
                    f"Could not load plugin {normalized!r}: {type(exc).__name__}: {exc}"
                )
                self._errors[normalized] = load_error
                raise load_error from exc
            self._manifests[normalized] = manifest
            return manifest

    def component(self, kind: PluginKind, reference: str) -> Callable[..., Any]:
        """Resolve one qualified component and validate its declaration."""

        parsed = parse_component_reference(reference)
        manifest = self.load(parsed.plugin)
        attribute = _KIND_ATTRIBUTES[kind]
        components = getattr(manifest, attribute)
        component = components.get(parsed.component)
        if component is None:
            choices = ", ".join(sorted(components)) or "none"
            raise PluginNotFoundError(
                f"Plugin {parsed.plugin!r} has no {kind} {parsed.component!r}; "
                f"available: {choices}"
            )
        return cast(Callable[..., Any], component)

    def source(self, reference: str, request: SourceRequest) -> FrameSourceExtension:
        """Construct and structurally validate one plugin frame source."""

        factory = self.component("source", reference)
        try:
            source = factory(request)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Source extension {reference!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            name = getattr(source, "name", None)
            read = getattr(source, "read", None)
            close = getattr(source, "close", None)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Source extension {reference!r} failed contract inspection: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(name, str) or not name:
            raise PluginContractError(
                f"Source extension {reference!r} returned an object without a name"
            )
        if not callable(read) or not callable(close):
            raise PluginContractError(
                f"Source extension {reference!r} must provide read() and close()"
            )
        return cast(FrameSourceExtension, source)

    def renderer(self, reference: str, request: RendererRequest) -> RendererExtension:
        """Construct and structurally validate one plugin renderer."""

        factory = self.component("renderer", reference)
        try:
            renderer = factory(request)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Renderer extension {reference!r} failed to initialize: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            render = getattr(renderer, "render", None)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Renderer extension {reference!r} failed contract inspection: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(render):
            raise PluginContractError(
                f"Renderer extension {reference!r} must provide render()"
            )
        return cast(RendererExtension, renderer)

    def transform(
        self,
        reference: str,
        source: Any,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke an explicitly selected transform with isolated errors."""

        transform = self.component("transform", reference)
        try:
            return transform(TransformRequest(source, options or {}))
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Transform extension {reference!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def export(
        self,
        reference: str,
        source: Any,
        destination: str | Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ExportReceipt:
        """Invoke an explicitly selected exporter and validate its receipt."""

        exporter = self.component("exporter", reference)
        request = ExportRequest(source, Path(destination), options or {})
        try:
            receipt = exporter(request)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            raise PluginExecutionError(
                f"Exporter extension {reference!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(receipt, ExportReceipt):
            raise PluginContractError(
                f"Exporter extension {reference!r} must return ExportReceipt"
            )
        return receipt

    def _info_for_manifest(
        self,
        identifier: str,
        manifest: PluginManifest,
        *,
        state: str = "ready",
    ) -> PluginInfo:
        point = self._points.get(identifier)
        distribution, installed_version = (
            _entry_point_distribution(point) if point is not None else (None, "")
        )
        return PluginInfo(
            identifier=identifier,
            name=manifest.name,
            version=manifest.version or installed_version,
            description=manifest.description,
            distribution=distribution,
            entry_point=str(getattr(point, "value", "")) or None,
            state=state,
            api_version=manifest.api_version,
            sources=tuple(sorted(manifest.sources)),
            renderers=tuple(sorted(manifest.renderers)),
            transforms=tuple(sorted(manifest.transforms)),
            exporters=tuple(sorted(manifest.exporters)),
        )

    def info(self, identifier: str, *, load: bool = True) -> PluginInfo:
        """Return metadata for one plugin, optionally importing it."""

        normalized = _normalize_identifier(identifier, label="plugin identifier")
        with self._lock:
            self._discover()
            if load:
                manifest = self.load(normalized)
                return self._info_for_manifest(normalized, manifest)
            if normalized in self._manifests:
                return self._info_for_manifest(normalized, self._manifests[normalized])
            if normalized in self._conflicts:
                return PluginInfo(
                    normalized,
                    normalized,
                    "",
                    "",
                    None,
                    None,
                    "conflict",
                    error=f"Multiple installed plugins use identifier {normalized!r}",
                )
            error = self._errors.get(normalized)
            if error is not None:
                return PluginInfo(
                    normalized,
                    normalized,
                    "",
                    "",
                    None,
                    None,
                    "error",
                    error=str(error),
                )
            point = self._points.get(normalized)
            if point is None:
                raise PluginNotFoundError(f"Plugin {normalized!r} is not installed")
            distribution, version = _entry_point_distribution(point)
            return PluginInfo(
                identifier=normalized,
                name=normalized,
                version=version,
                description="",
                distribution=distribution,
                entry_point=str(getattr(point, "value", "")) or None,
                state="discovered",
            )

    def inventory(self, *, load: bool = False) -> tuple[PluginInfo, ...]:
        """List installed metadata; optionally probe each plugin independently."""

        with self._lock:
            self._discover()
            identifiers = sorted(
                set(self._points)
                | set(self._conflicts)
                | set(self._manifests)
                | set(self._errors)
            )
        results: list[PluginInfo] = []
        for identifier in identifiers:
            if identifier == "<discovery>" or identifier.startswith("<invalid:"):
                results.append(
                    PluginInfo(
                        identifier,
                        identifier,
                        "",
                        "",
                        None,
                        None,
                        "error",
                        error=str(self._errors[identifier]),
                    )
                )
                continue
            if load:
                try:
                    results.append(self.info(identifier, load=True))
                except PluginError as exc:
                    results.append(
                        PluginInfo(
                            identifier,
                            identifier,
                            "",
                            "",
                            None,
                            None,
                            "error",
                            error=str(exc),
                        )
                    )
            else:
                results.append(self.info(identifier, load=False))
        return tuple(results)


_registry: PluginRegistry | None = None
_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """Return the process-wide registry without discovering until first use."""

    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = PluginRegistry()
    return _registry


def register_plugin(
    identifier: str,
    manifest: PluginManifest,
    *,
    replace: bool = False,
) -> None:
    """Register an in-process plugin in the default registry."""

    get_plugin_registry().register(identifier, manifest, replace=replace)


def validate_render_output(reference: str, output: object) -> RenderOutput:
    """Validate plugin output at the trust boundary before presentation."""

    if not isinstance(output, RenderOutput):
        raise PluginContractError(
            f"Renderer extension {reference!r} must return RenderOutput"
        )
    if not isinstance(output.text, str) or not output.text:
        raise PluginContractError(
            f"Renderer extension {reference!r} returned empty or non-text output"
        )
    valid_width = isinstance(output.width, int) and not isinstance(output.width, bool)
    valid_height = isinstance(output.height, int) and not isinstance(
        output.height, bool
    )
    if not valid_width or not valid_height or output.width < 1 or output.height < 1:
        raise PluginContractError(
            f"Renderer extension {reference!r} returned invalid dimensions"
        )
    if len(output.text.split("\n")) != output.height:
        raise PluginContractError(
            f"Renderer extension {reference!r} returned text whose row count "
            "does not match its height"
        )
    return output


__all__ = [
    "ComponentReference",
    "PluginCompatibilityError",
    "PluginConflictError",
    "PluginContractError",
    "PluginError",
    "PluginExecutionError",
    "PluginInfo",
    "PluginKind",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginRegistry",
    "get_plugin_registry",
    "is_plugin_reference",
    "parse_component_reference",
    "plugins_enabled",
    "register_plugin",
    "validate_render_output",
]
