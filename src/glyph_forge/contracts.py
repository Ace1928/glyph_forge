"""Versioned public contracts for still-image rendering.

The classes in this module deliberately avoid importing NumPy, Pillow, media
backends, or user configuration.  Applications can therefore validate and
serialize a render request without paying the cost of initializing the render
engine or touching the filesystem.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, cast

from .persistence import AtomicWriteError, atomic_write_bytes
from .visual_defaults import (
    DEFAULT_BRIGHTNESS,
    DEFAULT_CONTRAST,
    MAX_TONE,
    MIN_TONE,
)

RENDER_CONTRACT_VERSION = 1
MAX_CELL_DIMENSION = 4096
MAX_OUTPUT_DIMENSION = 32768
_RENDER_MODES = {"glyph", "edge", "braille", "half-block", "quadrant"}
_EDGE_ALGORITHMS = {"sobel", "prewitt", "scharr", "laplacian", "canny"}
_IDENTIFIER = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_PLUGIN_RENDER_MODE = re.compile(rf"plugin:{_IDENTIFIER}/{_IDENTIFIER}")


class GlyphForgeRenderError(Exception):
    """Base class for failures exposed by the canonical render pipeline."""


class RenderContractError(GlyphForgeRenderError, ValueError):
    """A render request is malformed or describes an unsupported combination."""


class SourceLoadError(GlyphForgeRenderError):
    """An input image could not be decoded into a renderable frame."""


class RenderExecutionError(GlyphForgeRenderError):
    """A validated render could not be completed."""


class RenderExportError(GlyphForgeRenderError):
    """A completed artifact could not be encoded or saved."""


class RenderFormat(str, Enum):
    """Encoding produced by a still-image render."""

    TEXT = "text"
    ANSI256 = "ansi256"
    TRUECOLOR = "truecolor"
    HTML = "html"
    PNG = "png"
    SVG = "svg"


class FitMode(str, Enum):
    """How glyph art is mapped onto an explicitly sized graphical canvas."""

    CONTAIN = "contain"
    COVER = "cover"
    STRETCH = "stretch"


class Alignment(str, Enum):
    """Anchor used when a contained or covered render has excess content."""

    TOP_LEFT = "top-left"
    TOP = "top"
    TOP_RIGHT = "top-right"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM = "bottom"
    BOTTOM_RIGHT = "bottom-right"


def _enum_value(value: Enum | str, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise RenderContractError(f"{name} must be a string")
    try:
        return enum_type(value.casefold())
    except ValueError as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise RenderContractError(
            f"Unknown {name} {value!r}; choose {choices}"
        ) from exc


def _positive_dimension(value: int | None, name: str, maximum: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise RenderContractError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise RenderContractError(f"{name} must be between 1 and {maximum}")


def _tone(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RenderContractError(f"{name} must be a number") from exc
    if not math.isfinite(numeric):
        raise RenderContractError(f"{name} must be a finite number")
    if not MIN_TONE <= numeric <= MAX_TONE:
        raise RenderContractError(
            f"{name} must be between {MIN_TONE:.1f} and {MAX_TONE:.1f}"
        )
    return numeric


def _integer_byte(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RenderContractError(f"{name} must be an integer")
    if not 0 <= value <= 255:
        raise RenderContractError(f"{name} must be between 0 and 255")
    return value


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise RenderContractError(f"{name} must be a number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RenderContractError(f"{name} must be a number") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise RenderContractError(f"{name} must be a positive finite number")
    return numeric


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Complete, serializable request for one still-image render.

    Cell dimensions describe artistic detail.  ``output_width`` and
    ``output_height`` describe only the final PNG/SVG canvas, keeping output
    pixels independent from glyph density.
    """

    width: int = 100
    height: int | None = None
    mode: str = "glyph"
    output_format: RenderFormat | str = RenderFormat.TEXT
    charset: str = "general"
    invert: bool = False
    dither: bool = False
    threshold: int = 128
    edge_algorithm: str = "sobel"
    edge_threshold: int = 48
    cell_aspect: float = 0.5
    resample: str = "bilinear"
    brightness: float = DEFAULT_BRIGHTNESS
    contrast: float = DEFAULT_CONTRAST
    style: str | None = None
    optimize: bool = False
    max_width: int | None = None
    max_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    fit: FitMode | str = FitMode.CONTAIN
    alignment: Alignment | str = Alignment.CENTER
    foreground: str = "#e8fff7"
    background: str = "#07110f"
    font: str | None = None
    contract_version: int = field(
        default=RENDER_CONTRACT_VERSION,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        self._validate_version()
        self._validate_dimensions()
        self._validate_text_fields()
        self._validate_boolean_fields()
        _integer_byte(self.threshold, "threshold")
        _integer_byte(self.edge_threshold, "edge_threshold")
        cell_aspect = _positive_float(self.cell_aspect, "cell_aspect")
        selected_format = _enum_value(
            self.output_format,
            RenderFormat,
            "output format",
        )
        selected_fit = _enum_value(self.fit, FitMode, "fit mode")
        selected_alignment = _enum_value(self.alignment, Alignment, "alignment")
        normalized_mode = self._normalized_mode()
        normalized_edge = self._normalized_edge_algorithm()

        object.__setattr__(self, "output_format", selected_format)
        object.__setattr__(self, "fit", selected_fit)
        object.__setattr__(self, "alignment", selected_alignment)
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "edge_algorithm", normalized_edge)
        object.__setattr__(self, "resample", self.resample.casefold())
        object.__setattr__(self, "cell_aspect", cell_aspect)
        object.__setattr__(self, "brightness", _tone(self.brightness, "brightness"))
        object.__setattr__(self, "contrast", _tone(self.contrast, "contrast"))
        self._validate_combinations(cast(RenderFormat, selected_format))

    def _validate_version(self) -> None:
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != RENDER_CONTRACT_VERSION
        ):
            raise RenderContractError(
                "Unsupported render contract version "
                f"{self.contract_version}; expected {RENDER_CONTRACT_VERSION}"
            )

    def _validate_dimensions(self) -> None:
        _positive_dimension(self.width, "width", MAX_CELL_DIMENSION)
        _positive_dimension(self.height, "height", MAX_CELL_DIMENSION)
        _positive_dimension(self.max_width, "max_width", MAX_CELL_DIMENSION)
        _positive_dimension(self.max_height, "max_height", MAX_CELL_DIMENSION)
        _positive_dimension(
            self.output_width,
            "output_width",
            MAX_OUTPUT_DIMENSION,
        )
        _positive_dimension(
            self.output_height,
            "output_height",
            MAX_OUTPUT_DIMENSION,
        )

    def _validate_text_fields(self) -> None:
        if not isinstance(self.charset, str) or not self.charset:
            raise RenderContractError("charset cannot be empty")
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise RenderContractError("mode cannot be empty")
        if not isinstance(self.edge_algorithm, str) or not self.edge_algorithm.strip():
            raise RenderContractError("edge_algorithm cannot be empty")
        if not isinstance(self.resample, str):
            raise RenderContractError("resample must be a string")
        if self.resample.casefold() not in {
            "nearest",
            "bilinear",
            "bicubic",
            "lanczos",
        }:
            raise RenderContractError(
                "resample must be nearest, bilinear, bicubic, or lanczos"
            )
        for name in ("foreground", "background"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RenderContractError(f"{name} must be a non-empty color string")
        if self.font is not None and (
            not isinstance(self.font, str) or not self.font.strip()
        ):
            raise RenderContractError("font must be a non-empty string when provided")
        if self.style is not None and not isinstance(self.style, str):
            raise RenderContractError("style must be a string when provided")

    def _validate_boolean_fields(self) -> None:
        for name in ("invert", "dither", "optimize"):
            if not isinstance(getattr(self, name), bool):
                raise RenderContractError(f"{name} must be true or false")

    def _normalized_mode(self) -> str:
        normalized_mode = self.mode.strip().casefold()
        if normalized_mode not in _RENDER_MODES and not _PLUGIN_RENDER_MODE.fullmatch(
            normalized_mode
        ):
            choices = ", ".join(sorted(_RENDER_MODES))
            raise RenderContractError(
                f"Unknown render mode {self.mode!r}; choose {choices}, or "
                "plugin:plugin-id/renderer"
            )
        return normalized_mode

    def _normalized_edge_algorithm(self) -> str:
        normalized_edge = self.edge_algorithm.strip().casefold()
        if normalized_edge not in _EDGE_ALGORITHMS:
            choices = ", ".join(sorted(_EDGE_ALGORITHMS))
            raise RenderContractError(
                f"Unknown edge algorithm {self.edge_algorithm!r}; choose {choices}"
            )
        return normalized_edge

    def _validate_combinations(self, selected_format: RenderFormat) -> None:
        graphical = selected_format in {RenderFormat.PNG, RenderFormat.SVG}
        if (
            self.output_width is not None or self.output_height is not None
        ) and not graphical:
            raise RenderContractError(
                "output_width and output_height require PNG or SVG output"
            )
        if selected_format is RenderFormat.HTML and self.mode != "glyph":
            raise RenderContractError("HTML color output currently requires glyph mode")
        if self.style and selected_format in {
            RenderFormat.ANSI256,
            RenderFormat.TRUECOLOR,
            RenderFormat.HTML,
        }:
            raise RenderContractError(
                "Text styles require plain text, PNG, or SVG output"
            )

    def with_updates(self, **updates: Any) -> "RenderRequest":
        """Return a validated copy with selected fields replaced."""

        return replace(self, **updates)

    @property
    def render_format(self) -> RenderFormat:
        """Normalized output format after request validation."""

        return cast(RenderFormat, self.output_format)

    @property
    def fit_mode(self) -> FitMode:
        """Normalized graphical fit policy after request validation."""

        return cast(FitMode, self.fit)

    @property
    def alignment_mode(self) -> Alignment:
        """Normalized graphical alignment after request validation."""

        return cast(Alignment, self.alignment)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation for presets/projects."""

        values = asdict(self)
        values["output_format"] = self.render_format.value
        values["fit"] = self.fit_mode.value
        values["alignment"] = self.alignment_mode.value
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RenderRequest":
        """Construct and validate a request from serialized data."""

        if not isinstance(values, Mapping):
            raise RenderContractError("Serialized render request must be an object")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise RenderContractError(
                f"Malformed serialized render request: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RenderMetrics:
    """Timing and geometry measurements for one completed render."""

    source_width: int
    source_height: int
    columns: int
    rows: int
    load_ms: float
    render_ms: float
    encode_ms: float
    total_ms: float
    output_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RenderArtifact:
    """Rendered glyph text plus its selected encoded representation."""

    request: RenderRequest
    glyph_text: str
    data: str | bytes
    media_type: str
    suffix: str
    columns: int
    rows: int
    pixel_width: int | None
    pixel_height: int | None
    metrics: RenderMetrics

    @property
    def is_binary(self) -> bool:
        return isinstance(self.data, bytes)

    @property
    def byte_size(self) -> int:
        if isinstance(self.data, bytes):
            return len(self.data)
        return len(self.data.encode("utf-8"))

    @property
    def text(self) -> str:
        """Plain glyph text, convenient for previews and compatibility callers."""

        return self.glyph_text

    def save(self, destination: str | os.PathLike[str]) -> Path:
        """Atomically save this artifact without exposing a partial destination."""

        payload = (
            self.data if isinstance(self.data, bytes) else self.data.encode("utf-8")
        )
        try:
            return atomic_write_bytes(destination, payload)
        except AtomicWriteError as exc:
            raise RenderExportError(str(exc)) from exc


__all__ = [
    "Alignment",
    "FitMode",
    "GlyphForgeRenderError",
    "MAX_CELL_DIMENSION",
    "MAX_OUTPUT_DIMENSION",
    "RENDER_CONTRACT_VERSION",
    "RenderArtifact",
    "RenderContractError",
    "RenderExecutionError",
    "RenderExportError",
    "RenderFormat",
    "RenderMetrics",
    "RenderRequest",
    "SourceLoadError",
]
