"""Vectorized live-frame renderers.

The hot path reduces source frames before producing Python strings.  This keeps
the amount of work proportional to the output surface rather than the source
resolution and performs pixel math in NumPy instead of per-pixel Python loops.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..runtime import RuntimeProfile, detect_runtime_profile
from ..utils.alphabet_manager import AlphabetManager
from .edges import EdgeAlgorithm, detect_edges, directional_glyphs, normalize_algorithm

RGBFrame = NDArray[np.uint8]


class RenderMode(str, Enum):
    """Spatial encoding used for each terminal character cell."""

    GLYPH = "glyph"
    EDGE = "edge"
    BRAILLE = "braille"
    HALF_BLOCK = "half-block"
    QUADRANT = "quadrant"


class ColorOutput(str, Enum):
    """Color encoding for terminal output."""

    NONE = "none"
    ANSI256 = "ansi256"
    TRUECOLOR = "truecolor"


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Immutable rendering options reusable across a live session."""

    width: int = 100
    height: int | None = None
    mode: RenderMode | str = RenderMode.GLYPH
    color: ColorOutput | str = ColorOutput.NONE
    charset: str = "general"
    invert: bool = False
    dither: bool = False
    threshold: int = 128
    edge_algorithm: EdgeAlgorithm | str = EdgeAlgorithm.SOBEL
    edge_threshold: int = 48
    cell_aspect: float = 0.5
    resample: str = "bilinear"

    @classmethod
    def adaptive(
        cls,
        preference: str = "auto",
        **overrides: Any,
    ) -> "RenderConfig":
        """Build a renderer configuration from a hardware profile."""

        profile = detect_runtime_profile(preference)
        values: dict[str, Any] = {
            "width": profile.stream_width,
            "resample": profile.resample,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """One rendered frame plus its logical terminal dimensions."""

    text: str
    width: int
    height: int
    mode: RenderMode


def _normalize_mode(mode: RenderMode | str) -> RenderMode:
    if isinstance(mode, RenderMode):
        return mode
    try:
        return RenderMode(mode.casefold())
    except ValueError as exc:
        choices = ", ".join(item.value for item in RenderMode)
        raise ValueError(f"Unknown render mode {mode!r}; choose {choices}") from exc


def _normalize_color(color: ColorOutput | str) -> ColorOutput:
    if isinstance(color, ColorOutput):
        return color
    aliases = {"ansi": ColorOutput.ANSI256, "rgb": ColorOutput.TRUECOLOR}
    value = color.casefold()
    if value in aliases:
        return aliases[value]
    try:
        return ColorOutput(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ColorOutput)
        raise ValueError(f"Unknown color mode {color!r}; choose {choices}") from exc


def _normalize_frame(frame: NDArray[Any]) -> RGBFrame:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(
            "Frames must have shape (height, width), (..., 3), or (..., 4)"
        )
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


@lru_cache(maxsize=1)
def _opencv() -> Any | None:
    try:
        import cv2  # type: ignore[import-untyped]

        return cv2
    except ImportError:
        return None


def _resize(frame: RGBFrame, width: int, height: int, method: str) -> RGBFrame:
    if width < 1 or height < 1:
        raise ValueError("Rendered width and height must be positive")
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame

    cv2 = _opencv()
    if cv2 is not None:
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(method, cv2.INTER_LINEAR)
        return np.asarray(
            cv2.resize(frame, (width, height), interpolation=interpolation),
            dtype=np.uint8,
        )

    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }.get(method, Image.Resampling.BILINEAR)
    return np.asarray(
        Image.fromarray(frame, mode="RGB").resize((width, height), resampling),
        dtype=np.uint8,
    )


def _grayscale(frame: RGBFrame) -> NDArray[np.uint8]:
    """Convert RGB to luma with integer SIMD-friendly arithmetic."""

    pixels = frame.astype(np.uint16, copy=False)
    values = (pixels[:, :, 0] * 77 + pixels[:, :, 1] * 150 + pixels[:, :, 2] * 29) >> 8
    return values.astype(np.uint8)


def _cell_height(frame: RGBFrame, config: RenderConfig) -> int:
    if config.height is not None:
        return max(1, config.height)
    aspect = frame.shape[0] / max(1, frame.shape[1])
    calculated = aspect * max(1, config.width) * config.cell_aspect
    return max(1, int(round(calculated)))


def _cell_dimensions(
    frame: RGBFrame,
    config: RenderConfig,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
) -> tuple[int, int]:
    """Fit the requested cell surface inside optional bounds without distortion."""

    if max_width is not None and max_width < 1:
        raise ValueError("max_width must be positive")
    if max_height is not None and max_height < 1:
        raise ValueError("max_height must be positive")

    width = config.width
    height = _cell_height(frame, config)
    width_scale = max_width / width if max_width is not None else 1.0
    height_scale = max_height / height if max_height is not None else 1.0
    scale = min(1.0, width_scale, height_scale)
    if scale >= 1.0:
        return width, height
    return max(1, int(width * scale)), max(1, int(height * scale))


def _ansi256(rgb: NDArray[np.uint8]) -> NDArray[np.uint16]:
    # Integer arithmetic is both faster and exactly equivalent to rounding an
    # 8-bit channel onto the six-level xterm colour cube.  Ellipsis keeps the
    # helper useful for both whole frames and individual rows.
    levels = (rgb.astype(np.uint16) * 5 + 127) // 255
    return 16 + levels[..., 0] * 36 + levels[..., 1] * 6 + levels[..., 2]


def _half_block_lines(
    upper: RGBFrame,
    lower: RGBFrame,
    mode: ColorOutput,
) -> list[str]:
    """Encode paired pixel rows with one vectorized colour conversion.

    Terminals retain the active foreground and background colours, so each
    row is emitted as runs.  In indexed mode the run comparison happens after
    quantization: RGB values which produce the same terminal colour no longer
    waste bandwidth on redundant escape sequences.
    """

    height, width, _ = upper.shape
    lines: list[str] = []
    block = "▀"

    if mode is ColorOutput.ANSI256:
        upper_indices = _ansi256(upper)
        lower_indices = _ansi256(lower)
        pairs = (upper_indices.astype(np.uint32) << 8) | lower_indices
        for y in range(height):
            row = pairs[y]
            starts = np.flatnonzero(
                np.concatenate((np.ones(1, dtype=np.bool_), row[1:] != row[:-1]))
            )
            ends = np.append(starts[1:], width)
            parts: list[str] = []
            for start, end in zip(starts, ends, strict=True):
                foreground = int(upper_indices[y, start])
                background = int(lower_indices[y, start])
                parts.append(f"\x1b[38;5;{foreground};48;5;{background}m")
                parts.append(block * int(end - start))
            parts.append("\x1b[0m")
            lines.append("".join(parts))
        return lines

    for y in range(height):
        top = upper[y]
        bottom = lower[y]
        changed = np.empty(width, dtype=np.bool_)
        changed[0] = True
        changed[1:] = np.logical_or(
            np.any(top[1:] != top[:-1], axis=1),
            np.any(bottom[1:] != bottom[:-1], axis=1),
        )
        starts = np.flatnonzero(changed)
        ends = np.append(starts[1:], width)
        parts = []
        for start, end in zip(starts, ends, strict=True):
            foreground = top[start]
            background = bottom[start]
            parts.append(
                f"\x1b[38;2;{int(foreground[0])};{int(foreground[1])};"
                f"{int(foreground[2])};48;2;{int(background[0])};"
                f"{int(background[1])};{int(background[2])}m"
            )
            parts.append(block * int(end - start))
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return lines


def _colorize_rows(
    rows: list[str],
    colors: RGBFrame,
    mode: ColorOutput,
) -> str:
    if mode is ColorOutput.NONE:
        return "\n".join(rows)

    indexed = _ansi256(colors) if mode is ColorOutput.ANSI256 else None
    output: list[str] = []
    for y, row in enumerate(rows):
        parts: list[str] = []
        previous: Any = None
        for x, character in enumerate(row):
            if mode is ColorOutput.ANSI256:
                current: Any = int(indexed[y, x])  # type: ignore[index]
                if current != previous:
                    parts.append(f"\x1b[38;5;{current}m")
            else:
                current = tuple(int(value) for value in colors[y, x])
                if current != previous:
                    parts.append(f"\x1b[38;2;{current[0]};{current[1]};{current[2]}m")
            parts.append(character)
            previous = current
        parts.append("\x1b[0m")
        output.append("".join(parts))
    return "\n".join(output)


_BAYER_4 = np.asarray(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.uint8,
)


def _binary_pixels(gray: NDArray[np.uint8], config: RenderConfig) -> NDArray[np.bool_]:
    if config.dither:
        tiled = np.tile(
            _BAYER_4,
            (
                (gray.shape[0] + 3) // 4,
                (gray.shape[1] + 3) // 4,
            ),
        )[: gray.shape[0], : gray.shape[1]]
        result = gray > (tiled.astype(np.uint16) * 16 + 7)
    else:
        result = gray >= max(0, min(255, config.threshold))
    return np.logical_not(result) if config.invert else result


class FrameRenderer:
    """Reusable dispatcher for vectorized glyph and subcell renderers."""

    def __init__(
        self,
        config: RenderConfig | None = None,
        *,
        profile: RuntimeProfile | None = None,
    ) -> None:
        if config is None:
            selected = profile or detect_runtime_profile()
            config = RenderConfig(
                width=selected.stream_width,
                resample=selected.resample,
            )
        if config.width < 1:
            raise ValueError("Rendered width must be positive")
        if config.cell_aspect <= 0:
            raise ValueError("cell_aspect must be positive")
        if not 0 <= config.edge_threshold <= 255:
            raise ValueError("edge_threshold must be between 0 and 255")
        self.config = config
        self.mode = _normalize_mode(config.mode)
        self.color = _normalize_color(config.color)
        self.edge_algorithm = normalize_algorithm(config.edge_algorithm)
        self.charset = (
            AlphabetManager.get_alphabet(config.charset)
            if config.charset in AlphabetManager.list_available_alphabets()
            else config.charset
        )
        if not self.charset:
            raise ValueError("charset cannot be empty")
        if config.invert:
            self.charset = self.charset[::-1]
        self._glyphs = np.asarray(tuple(self.charset), dtype="<U1")

    def render(
        self,
        frame: NDArray[Any],
        *,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> RenderResult:
        """Render a frame, optionally fitting it inside a terminal cell surface."""

        normalized = _normalize_frame(frame)
        width, height = _cell_dimensions(
            normalized,
            self.config,
            max_width=max_width,
            max_height=max_height,
        )
        if self.mode is RenderMode.BRAILLE:
            return self._render_braille(normalized, width, height)
        if self.mode is RenderMode.EDGE:
            return self._render_edge(normalized, width, height)
        if self.mode is RenderMode.HALF_BLOCK:
            return self._render_half_block(normalized, width, height)
        if self.mode is RenderMode.QUADRANT:
            return self._render_quadrant(normalized, width, height)
        return self._render_glyph(normalized, width, height)

    def _render_glyph(self, frame: RGBFrame, width: int, height: int) -> RenderResult:
        resized = _resize(frame, width, height, self.config.resample)
        gray = _grayscale(resized)
        indices = (gray.astype(np.uint16) * len(self._glyphs) // 256).clip(
            max=len(self._glyphs) - 1
        )
        mapped = self._glyphs[indices]
        rows = ["".join(row) for row in mapped.tolist()]
        text = _colorize_rows(rows, resized, self.color)
        return RenderResult(text, width, height, self.mode)

    def _render_edge(self, frame: RGBFrame, width: int, height: int) -> RenderResult:
        resized = _resize(frame, width, height, self.config.resample)
        gray = _grayscale(resized)
        indices = (gray.astype(np.uint16) * len(self._glyphs) // 256).clip(
            max=len(self._glyphs) - 1
        )
        mapped = self._glyphs[indices].copy()
        edges = detect_edges(
            gray,
            self.edge_algorithm,
            threshold=self.config.edge_threshold,
        )
        edge_mask = edges.magnitude > 0
        mapped[edge_mask] = directional_glyphs(edges)[edge_mask]
        rows = ["".join(row) for row in mapped.tolist()]
        text = _colorize_rows(rows, resized, self.color)
        return RenderResult(text, width, height, self.mode)

    def _render_braille(self, frame: RGBFrame, width: int, height: int) -> RenderResult:
        resized = _resize(
            frame,
            width * 2,
            height * 4,
            self.config.resample,
        )
        on = _binary_pixels(_grayscale(resized), self.config)
        blocks = on.reshape(height, 4, width, 2)
        codes = (
            blocks[:, 0, :, 0].astype(np.uint16)
            | (blocks[:, 1, :, 0].astype(np.uint16) << 1)
            | (blocks[:, 2, :, 0].astype(np.uint16) << 2)
            | (blocks[:, 0, :, 1].astype(np.uint16) << 3)
            | (blocks[:, 1, :, 1].astype(np.uint16) << 4)
            | (blocks[:, 2, :, 1].astype(np.uint16) << 5)
            | (blocks[:, 3, :, 0].astype(np.uint16) << 6)
            | (blocks[:, 3, :, 1].astype(np.uint16) << 7)
        )
        rows = ["".join(chr(0x2800 + int(code)) for code in row) for row in codes]
        colors = (
            resized.reshape(height, 4, width, 2, 3)
            .mean(axis=(1, 3), dtype=np.float32)
            .astype(np.uint8)
        )
        text = _colorize_rows(rows, colors, self.color)
        return RenderResult(text, width, height, self.mode)

    def _render_half_block(
        self,
        frame: RGBFrame,
        width: int,
        height: int,
    ) -> RenderResult:
        resized = _resize(
            frame,
            width,
            height * 2,
            self.config.resample,
        )
        if self.color is ColorOutput.NONE:
            fallback = replace(self.config, mode=RenderMode.GLYPH)
            return FrameRenderer(fallback).render(
                frame,
                max_width=width,
                max_height=height,
            )

        upper = resized[0::2]
        lower = resized[1::2]
        lines = _half_block_lines(upper, lower, self.color)
        return RenderResult("\n".join(lines), width, height, self.mode)

    def _render_quadrant(
        self, frame: RGBFrame, width: int, height: int
    ) -> RenderResult:
        resized = _resize(
            frame,
            width * 2,
            height * 2,
            self.config.resample,
        )
        on = _binary_pixels(_grayscale(resized), self.config)
        blocks = on.reshape(height, 2, width, 2)
        masks = (
            blocks[:, 0, :, 0].astype(np.uint8)
            | (blocks[:, 0, :, 1].astype(np.uint8) << 1)
            | (blocks[:, 1, :, 0].astype(np.uint8) << 2)
            | (blocks[:, 1, :, 1].astype(np.uint8) << 3)
        )
        symbols = np.asarray(tuple(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"), dtype="<U1")
        rows = ["".join(row) for row in symbols[masks].tolist()]
        colors = (
            resized.reshape(height, 2, width, 2, 3)
            .mean(axis=(1, 3), dtype=np.float32)
            .astype(np.uint8)
        )
        text = _colorize_rows(rows, colors, self.color)
        return RenderResult(text, width, height, self.mode)


def render_svg(
    frame: NDArray[Any],
    config: RenderConfig | None = None,
    *,
    font_size: float = 10.0,
    foreground: str = "#e8fff7",
    background: str = "#07110f",
    font_family: str = "ui-monospace, SFMono-Regular, Consolas, monospace",
) -> str:
    """Render a still frame as scalable SVG text.

    ANSI color is intentionally disabled for SVG output.  The result contains
    real text glyphs and remains sharp at arbitrary zoom levels.
    """

    selected = config or RenderConfig()
    result = FrameRenderer(replace(selected, color=ColorOutput.NONE)).render(frame)
    line_height = font_size * 1.05
    width = max(1.0, result.width * font_size * 0.62)
    height = max(line_height, result.height * line_height)
    rows = []
    for index, line in enumerate(result.text.splitlines(), start=1):
        rows.append(
            f'<text x="0" y="{index * line_height:.2f}">{html.escape(line)}</text>'
        )
    family = html.escape(font_family, quote=True)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}" '
        f'height="{height:.2f}" viewBox="0 0 {width:.2f} {height:.2f}" '
        'role="img" aria-label="Glyph Forge render">\n'
        f'<rect width="100%" height="100%" fill="{html.escape(background)}"/>\n'
        f'<g fill="{html.escape(foreground)}" font-family="{family}" '
        f'font-size="{font_size:.2f}" xml:space="preserve">\n'
        + "\n".join(rows)
        + "\n</g>\n</svg>\n"
    )


__all__ = [
    "ColorOutput",
    "FrameRenderer",
    "RenderConfig",
    "RenderMode",
    "RenderResult",
    "render_svg",
]
