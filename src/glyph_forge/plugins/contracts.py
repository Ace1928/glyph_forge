"""Stable contracts for third-party Glyph Forge extensions.

This module intentionally depends only on the Python standard library. Plugin
packages can import it without loading NumPy, Pillow, OpenCV, or either UI.
Concrete media types remain deliberately structural so optional dependencies
stay optional and the contract can evolve without coupling extensions to an
implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Protocol, runtime_checkable

PLUGIN_API_VERSION: Final[int] = 1
PLUGIN_ENTRY_POINT_GROUP: Final[str] = "glyph_forge.plugins"


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """Parameters supplied when a plugin frame source is opened."""

    resource: str = ""
    width: int | None = None
    height: int | None = None
    fps: float = 30.0
    loop: bool = False


@dataclass(frozen=True, slots=True)
class RendererRequest:
    """Parameters supplied once when a plugin renderer is constructed."""

    reference: str
    config: Any


@dataclass(frozen=True, slots=True)
class RenderOutput:
    """Renderer-neutral text surface returned by a plugin renderer."""

    text: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class TransformRequest:
    """Input and options supplied to an explicitly invoked transform."""

    source: Any
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Input, destination, and options supplied to a plugin exporter."""

    source: Any
    destination: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", Path(self.destination))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class ExportReceipt:
    """Portable result returned by a successful plugin exporter."""

    output: Path
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class FrameSourceExtension(Protocol):
    """Structural source contract shared with the live capture engine."""

    @property
    def name(self) -> str: ...

    def read(self) -> Any | None: ...

    def close(self) -> None: ...


@runtime_checkable
class RendererExtension(Protocol):
    """A reusable plugin renderer constructed once per session."""

    def render(
        self,
        frame: Any,
        *,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> RenderOutput: ...


SourceFactory = Callable[[SourceRequest], FrameSourceExtension]
RendererFactory = Callable[[RendererRequest], RendererExtension]
TransformExtension = Callable[[TransformRequest], Any]
ExporterExtension = Callable[[ExportRequest], ExportReceipt]


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Versioned declaration exported through ``glyph_forge.plugins``.

    The installed entry-point name is the plugin's stable identifier. ``name``
    is a human-readable label and may contain spaces. Component names are local
    to the plugin and are addressed as ``plugin-id/component``.
    """

    name: str
    version: str
    description: str = ""
    api_version: int = PLUGIN_API_VERSION
    sources: Mapping[str, SourceFactory] = field(default_factory=dict)
    renderers: Mapping[str, RendererFactory] = field(default_factory=dict)
    transforms: Mapping[str, TransformExtension] = field(default_factory=dict)
    exporters: Mapping[str, ExporterExtension] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for attribute in ("sources", "renderers", "transforms", "exporters"):
            values = dict(getattr(self, attribute))
            object.__setattr__(self, attribute, MappingProxyType(values))


__all__ = [
    "PLUGIN_API_VERSION",
    "PLUGIN_ENTRY_POINT_GROUP",
    "ExportReceipt",
    "ExportRequest",
    "ExporterExtension",
    "FrameSourceExtension",
    "PluginManifest",
    "RenderOutput",
    "RendererExtension",
    "RendererFactory",
    "RendererRequest",
    "SourceFactory",
    "SourceRequest",
    "TransformExtension",
    "TransformRequest",
]
