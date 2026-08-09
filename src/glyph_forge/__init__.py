"""Glyph Forge's stable, lightweight public API.

Importing :mod:`glyph_forge` performs no filesystem writes and does not load
media or UI backends.  Public implementations are imported on first use so
command discovery, diagnostics, and partial installations remain responsive.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import sys
from importlib import import_module
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

VERSION: Final[Tuple[int, int, int]] = (0, 2, 0)
__version__ = ".".join(map(str, VERSION))
__author__ = "Lloyd Handyside"
__license__ = "MIT"
__maintainer__ = "Neuroforge"
__email__ = "ace1928@gmail.com"
__maintainer_email__ = "lloyd.handyside@neuroforge.io"
__status__ = "Beta"
__copyright__ = "Copyright 2023-2026 Neuroforge"

logging.getLogger(__name__).addHandler(logging.NullHandler())


class MaintainerInfo(TypedDict):
    """Maintainer contact details."""

    name: str
    email: str


class ProjectInfo(TypedDict):
    """Public project metadata."""

    name: str
    description: str
    version: Tuple[int, int, int]
    author: str
    email: str
    organization: str
    org_email: str
    url: str
    license: str
    copyright: str
    maintainers: List[MaintainerInfo]
    repository: str
    status: str


PROJECT: Final[ProjectInfo] = {
    "name": "Glyph Forge",
    "description": "Fast, portable image, text, and video-to-glyph art toolkit",
    "version": VERSION,
    "author": __author__,
    "email": __email__,
    "organization": "Neuroforge",
    "org_email": __maintainer_email__,
    "url": "https://github.com/Ace1928/glyph_forge",
    "license": __license__,
    "copyright": __copyright__,
    "maintainers": [
        {"name": __maintainer__, "email": __maintainer_email__},
        {"name": "Eidos", "email": "syntheticeidos@gmail.com"},
    ],
    "repository": "https://github.com/Ace1928/glyph_forge",
    "status": __status__,
}

TransformerMap = Dict[str, Callable[[bytes], bytes]]
RenderOptions = Dict[str, Union[str, int, float, bool]]
GlyphMatrix = List[List[str]]
T_co = TypeVar("T_co", covariant=True)

ColorMode = Literal["none", "ansi16", "ansi256", "truecolor", "rgb", "web"]
DitherAlgorithm = Literal[
    "none",
    "floyd-steinberg",
    "jarvis",
    "stucki",
    "atkinson",
    "burkes",
    "sierra",
]


class Renderer(Protocol[T_co]):
    """Contract implemented by output renderers."""

    def render(self, matrix: GlyphMatrix, options: RenderOptions) -> T_co: ...


class Transformer(Protocol):
    """Contract implemented by media transformers."""

    def transform(self, source: Any, **options: Any) -> GlyphMatrix: ...


class SystemCapabilities(TypedDict, total=False):
    """Portable runtime and feature information."""

    color_support: Dict[str, Any]
    terminal_size: Tuple[int, int]
    python_version: tuple[int, ...]
    platform: str
    glyph_forge_version: str
    has_pillow: bool
    has_numpy: bool
    has_rich: bool
    unicode_support: bool
    performance_profile: Dict[str, Any]
    features: List[Dict[str, Any]]


DEFAULT_CONFIG: Final[Dict[str, Any]] = {
    "char_sets": {
        "standard": " .:-=+*#%@",
        "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        "block": " ░▒▓█",
        "minimal": " ._|/\\#",
        "eidosian": "⚡✧✦⚛⚘⚔⚙⚚⚜⛭⛯❄❈❉❊",
    },
    "color_modes": ["none", "ansi16", "ansi256", "truecolor", "rgb", "web"],
    "default_width": 80,
    "default_height": 24,
    "dither_algorithms": [
        "none",
        "floyd-steinberg",
        "jarvis",
        "stucki",
        "atkinson",
        "burkes",
        "sierra",
    ],
    "edge_detection": True,
    "structure_path": {
        "temp": Path.home() / ".glyph_forge" / "temp",
        "cache": Path.home() / ".glyph_forge" / "cache",
        "output": Path.home() / ".glyph_forge" / "output",
        "resources": Path(__file__).parent / "resources",
        "config": Path.home() / ".glyph_forge" / "config",
    },
}

_PROFILE_OVERRIDES: Final[Dict[str, Dict[str, Any]]] = {
    "minimal": {"charset": "minimal", "optimization_level": 1},
    "standard": {"charset": "standard", "optimization_level": 2},
    "detailed": {"charset": "detailed", "optimization_level": 3},
    "eidosian": {
        "charset": "eidosian",
        "optimization_level": 4,
        "entropy_preservation": True,
    },
}


def get_config(profile: str | None = None) -> Dict[str, Any]:
    """Return an isolated configuration with profile and environment overrides."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    selected = _PROFILE_OVERRIDES.get(profile or "standard")
    if profile is not None and selected is None:
        choices = ", ".join(sorted(_PROFILE_OVERRIDES))
        raise ValueError(f"Unknown profile {profile!r}; choose one of: {choices}")
    assert selected is not None
    config["char_sets"]["active"] = config["char_sets"][selected["charset"]]
    config.update({key: value for key, value in selected.items() if key != "charset"})

    for key, value in tuple(config.items()):
        env_value = os.environ.get(f"GLYPH_FORGE_{key.upper()}")
        if env_value is None:
            continue
        if isinstance(value, bool):
            config[key] = env_value.casefold() in {"1", "true", "yes", "on"}
        elif isinstance(value, int):
            config[key] = int(env_value)
        elif isinstance(value, float):
            config[key] = float(env_value)
        elif isinstance(value, list):
            config[key] = [item.strip() for item in env_value.split(",")]
        else:
            config[key] = env_value
    return config


def get_project_info() -> ProjectInfo:
    """Return a defensive copy of public project metadata."""

    return copy.deepcopy(PROJECT)


def get_system_capabilities() -> SystemCapabilities:
    """Return terminal, dependency, and hardware-adaptive runtime details."""

    from .runtime import runtime_report

    report = runtime_report()
    terminal = shutil.get_terminal_size(fallback=(80, 24))
    capabilities = {item["key"]: item["available"] for item in report["capabilities"]}
    try:
        from .utils import detect_capabilities

        colors = detect_capabilities()
    except (ImportError, OSError):
        colors = {}
    return {
        "color_support": colors,
        "terminal_size": (terminal.columns, terminal.lines),
        "python_version": tuple(sys.version_info[:3]),
        "platform": sys.platform,
        "glyph_forge_version": __version__,
        "has_pillow": bool(capabilities.get("PIL")),
        "has_numpy": bool(capabilities.get("numpy")),
        "has_rich": bool(capabilities.get("rich")),
        "unicode_support": bool(
            sys.stdout.encoding and "utf" in sys.stdout.encoding.casefold()
        ),
        "performance_profile": report["profile"],
        "features": report["capabilities"],
    }


_LAZY_EXPORTS: Final[Dict[str, Tuple[str, str]]] = {
    "get_api": ("glyph_forge.api", "get_api"),
    "GlyphForgeAPI": ("glyph_forge.api", "GlyphForgeAPI"),
    "TextRenderer": ("glyph_forge.renderers", "TextRenderer"),
    "HTMLRenderer": ("glyph_forge.renderers", "HTMLRenderer"),
    "ANSIRenderer": ("glyph_forge.renderers", "ANSIRenderer"),
    "SVGRenderer": ("glyph_forge.renderers", "SVGRenderer"),
    "PluginManifest": ("glyph_forge.plugins", "PluginManifest"),
    "PluginRegistry": ("glyph_forge.plugins", "PluginRegistry"),
    "PluginInfo": ("glyph_forge.plugins", "PluginInfo"),
    "PluginError": ("glyph_forge.plugins", "PluginError"),
    "RenderOutput": ("glyph_forge.plugins", "RenderOutput"),
    "SourceRequest": ("glyph_forge.plugins", "SourceRequest"),
    "RendererRequest": ("glyph_forge.plugins", "RendererRequest"),
    "TransformRequest": ("glyph_forge.plugins", "TransformRequest"),
    "ExportRequest": ("glyph_forge.plugins", "ExportRequest"),
    "ExportReceipt": ("glyph_forge.plugins", "ExportReceipt"),
    "get_plugin_registry": ("glyph_forge.plugins", "get_plugin_registry"),
    "register_plugin": ("glyph_forge.plugins", "register_plugin"),
    "ImageTransformer": ("glyph_forge.transformers", "ImageTransformer"),
    "ColorMapper": ("glyph_forge.transformers", "ColorMapper"),
    "DepthAnalyzer": ("glyph_forge.transformers", "DepthAnalyzer"),
    "EdgeDetector": ("glyph_forge.transformers", "EdgeDetector"),
    "setup_logger": ("glyph_forge.utils", "setup_logger"),
    "configure": ("glyph_forge.utils", "configure"),
    "measure_performance": ("glyph_forge.utils", "measure_performance"),
    "detect_capabilities": ("glyph_forge.utils", "detect_capabilities"),
    "image_to_glyph": (
        "glyph_forge.services.image_to_glyph",
        "image_to_glyph",
    ),
    "text_to_banner": ("glyph_forge.services", "text_to_banner"),
    "video_to_glyph_frames": ("glyph_forge.services", "video_to_glyph_frames"),
    "iter_video_glyph_frames": (
        "glyph_forge.services",
        "iter_video_glyph_frames",
    ),
    "iter_video_images": ("glyph_forge.services", "iter_video_images"),
    "CaptureRegion": ("glyph_forge.live", "CaptureRegion"),
    "VideoExportConfig": ("glyph_forge.live", "VideoExportConfig"),
    "export_glyph_video": ("glyph_forge.live", "export_glyph_video"),
    "FrameRenderer": ("glyph_forge.live", "FrameRenderer"),
    "PluginRenderMode": ("glyph_forge.live", "PluginRenderMode"),
    "InputRouter": ("glyph_forge.live", "InputRouter"),
    "KeyInput": ("glyph_forge.live", "KeyInput"),
    "LatestFramePump": ("glyph_forge.live", "LatestFramePump"),
    "PointerInput": ("glyph_forge.live", "PointerInput"),
    "RenderConfig": ("glyph_forge.live", "RenderConfig"),
    "RenderMode": ("glyph_forge.live", "RenderMode"),
    "TerminalPresenter": ("glyph_forge.live", "TerminalPresenter"),
    "TerminalRedraw": ("glyph_forge.live", "TerminalRedraw"),
    "TerminalSessionConfig": ("glyph_forge.live", "TerminalSessionConfig"),
    "create_frame_source": ("glyph_forge.live", "create_frame_source"),
    "run_terminal_session": ("glyph_forge.live", "run_terminal_session"),
    "StudioServer": ("glyph_forge.studio", "StudioServer"),
    "SharePublication": ("glyph_forge.studio", "SharePublication"),
    "load_eidos_profile": ("glyph_forge.eidos_profile", "load_profile"),
    "save_eidos_profile": ("glyph_forge.eidos_profile", "save_profile"),
    "update_eidos_profile": ("glyph_forge.eidos_profile", "update_profile"),
    "EidosProfile": ("glyph_forge.eidos_profile", "EidosProfile"),
}


def __getattr__(name: str) -> Any:
    """Load substantial public components only when requested."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ANSIRenderer",
    "CaptureRegion",
    "ColorMapper",
    "ColorMode",
    "DEFAULT_CONFIG",
    "DepthAnalyzer",
    "DitherAlgorithm",
    "EdgeDetector",
    "EidosProfile",
    "FrameRenderer",
    "GlyphForgeAPI",
    "GlyphMatrix",
    "HTMLRenderer",
    "ImageTransformer",
    "InputRouter",
    "KeyInput",
    "LatestFramePump",
    "PointerInput",
    "PluginError",
    "PluginInfo",
    "PluginManifest",
    "PluginRegistry",
    "PluginRenderMode",
    "PROJECT",
    "RenderOptions",
    "RenderConfig",
    "RenderMode",
    "RenderOutput",
    "RendererRequest",
    "Renderer",
    "SVGRenderer",
    "SystemCapabilities",
    "TerminalPresenter",
    "TerminalRedraw",
    "TerminalSessionConfig",
    "StudioServer",
    "SharePublication",
    "SourceRequest",
    "TextRenderer",
    "Transformer",
    "TransformerMap",
    "TransformRequest",
    "VERSION",
    "VideoExportConfig",
    "ExportReceipt",
    "ExportRequest",
    "configure",
    "create_frame_source",
    "detect_capabilities",
    "get_api",
    "get_config",
    "get_project_info",
    "get_plugin_registry",
    "get_system_capabilities",
    "image_to_glyph",
    "iter_video_glyph_frames",
    "iter_video_images",
    "load_eidos_profile",
    "measure_performance",
    "register_plugin",
    "save_eidos_profile",
    "setup_logger",
    "text_to_banner",
    "run_terminal_session",
    "update_eidos_profile",
    "export_glyph_video",
    "video_to_glyph_frames",
    "__author__",
    "__email__",
    "__license__",
    "__version__",
]
