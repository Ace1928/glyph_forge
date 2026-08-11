"""Canonical still-image pipeline shared by every Glyph Forge interface."""

from __future__ import annotations

import html
import os
import re
from io import BytesIO
from pathlib import Path
from time import perf_counter_ns
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageColor, ImageOps, UnidentifiedImageError

from .contracts import (
    RenderArtifact,
    RenderContractError,
    RenderExecutionError,
    RenderExportError,
    RenderFormat,
    RenderMetrics,
    RenderRequest,
    SourceLoadError,
)
from .core.style_manager import apply_style
from .live.renderers import (
    ColorOutput,
    FrameRenderer,
    RenderConfig,
    render_text_png,
    render_text_svg,
)
from .utils.alphabet_manager import AlphabetManager

ImageArray: TypeAlias = NDArray[Any]
ImageSource: TypeAlias = str | os.PathLike[str] | Image.Image | ImageArray
MAX_SOURCE_PIXELS = 100_000_000

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRUECOLOR_ESCAPE = re.compile(r"\x1b\[38;2;(\d{1,3});(\d{1,3});(\d{1,3})m|\x1b\[0m")


def format_for_path(
    destination: str | os.PathLike[str] | None,
    *,
    color: str = "none",
) -> RenderFormat:
    """Infer a canonical output format from a path and legacy color option."""

    selected = color.casefold()
    if destination is not None:
        suffix = Path(destination).suffix.casefold()
        if suffix == ".png":
            return RenderFormat.PNG
        if suffix == ".svg":
            return RenderFormat.SVG
        if suffix in {".html", ".htm"}:
            return RenderFormat.HTML
        if suffix == ".ansi":
            if selected == "ansi256":
                return RenderFormat.ANSI256
            return RenderFormat.TRUECOLOR
    aliases = {
        "none": RenderFormat.TEXT,
        "ansi": RenderFormat.TRUECOLOR,
        "truecolor": RenderFormat.TRUECOLOR,
        "ansi256": RenderFormat.ANSI256,
        "html": RenderFormat.HTML,
    }
    try:
        return aliases[selected]
    except KeyError as exc:
        raise RenderContractError(
            f"Unknown color output {color!r}; choose none, ansi, ansi256, or html"
        ) from exc


def strip_ansi(text: str) -> str:
    """Remove terminal control sequences from rendered glyph text."""

    return _ANSI_ESCAPE.sub("", text)


def truecolor_to_html(text: str) -> str:
    """Convert renderer-owned truecolor ANSI runs into safe standalone HTML."""

    parts = ["<pre style='line-height:1; letter-spacing:0'>"]
    position = 0
    active: tuple[int, int, int] | None = None
    for match in _TRUECOLOR_ESCAPE.finditer(text):
        segment = text[position : match.start()]
        if segment:
            escaped = html.escape(segment)
            if active is None:
                parts.append(escaped)
            else:
                parts.append(
                    f"<span style='color:#{active[0]:02x}{active[1]:02x}"
                    f"{active[2]:02x}'>{escaped}</span>"
                )
        if match.group(1) is None:
            active = None
        else:
            active = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        position = match.end()
    tail = text[position:]
    if tail:
        escaped = html.escape(tail)
        if active is None:
            parts.append(escaped)
        else:
            parts.append(
                f"<span style='color:#{active[0]:02x}{active[1]:02x}"
                f"{active[2]:02x}'>{escaped}</span>"
            )
    parts.append("</pre>")
    return "".join(parts)


def _background_rgb(request: RenderRequest) -> tuple[int, int, int]:
    try:
        selected = ImageColor.getrgb(request.background)
        return selected[0], selected[1], selected[2]
    except ValueError as exc:
        raise RenderContractError(f"Invalid background color: {exc}") from exc


def _validate_colors(request: RenderRequest) -> None:
    for name in ("foreground", "background"):
        try:
            ImageColor.getrgb(getattr(request, name))
        except ValueError as exc:
            raise RenderContractError(f"Invalid {name} color: {exc}") from exc


def _composite_image(image: Image.Image, request: RenderRequest) -> Image.Image:
    transposed = ImageOps.exif_transpose(image)
    if transposed.width * transposed.height > MAX_SOURCE_PIXELS:
        raise SourceLoadError(
            "Source image exceeds the safe 100-megapixel decode budget"
        )
    if "A" in transposed.getbands() or "transparency" in transposed.info:
        foreground = transposed.convert("RGBA")
        background = Image.new(
            "RGBA", foreground.size, (*_background_rgb(request), 255)
        )
        return Image.alpha_composite(background, foreground).convert("RGB")
    return transposed.convert("RGB")


def _array_image(source: ImageArray, request: RenderRequest) -> Image.Image:
    array = np.asarray(source)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise SourceLoadError(
            "Image arrays must have shape (height, width), (..., 3), or (..., 4)"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise SourceLoadError("Source image cannot be empty")
    if int(array.shape[0]) * int(array.shape[1]) > MAX_SOURCE_PIXELS:
        raise SourceLoadError(
            "Source image exceeds the safe 100-megapixel decode budget"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    mode = "RGBA" if array.shape[2] == 4 else "RGB"
    return _composite_image(Image.fromarray(array, mode=mode), request)


def load_image(source: ImageSource, request: RenderRequest) -> Image.Image:
    """Decode, orient, and alpha-composite a supported image source."""

    try:
        if isinstance(source, Image.Image):
            return _composite_image(source, request)
        if isinstance(source, np.ndarray):
            return _array_image(source, request)
        if isinstance(source, (str, os.PathLike)):
            path = Path(source).expanduser()
            with Image.open(path) as opened:
                if opened.width * opened.height > MAX_SOURCE_PIXELS:
                    raise SourceLoadError(
                        "Source image exceeds the safe 100-megapixel decode budget"
                    )
                opened.load()
                return _composite_image(opened, request)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise SourceLoadError(f"Could not load image {source!s}: {exc}") from exc
    raise SourceLoadError(
        "Image source must be a path, Pillow image, or NumPy pixel array"
    )


def _renderer_color(output_format: RenderFormat) -> ColorOutput:
    if output_format is RenderFormat.ANSI256:
        return ColorOutput.ANSI256
    if output_format in {RenderFormat.TRUECOLOR, RenderFormat.HTML}:
        return ColorOutput.TRUECOLOR
    return ColorOutput.NONE


def _encoded_output(
    glyph_text: str,
    rendered_text: str,
    request: RenderRequest,
) -> tuple[str | bytes, str, str, int | None, int | None]:
    output_format = request.render_format
    if output_format is RenderFormat.TEXT:
        return glyph_text, "text/plain; charset=utf-8", ".txt", None, None
    if output_format in {RenderFormat.ANSI256, RenderFormat.TRUECOLOR}:
        return rendered_text, "text/plain; charset=utf-8", ".ansi", None, None
    if output_format is RenderFormat.HTML:
        document = truecolor_to_html(rendered_text)
        return document, "text/html; charset=utf-8", ".html", None, None
    if output_format is RenderFormat.SVG:
        try:
            document = render_text_svg(
                glyph_text,
                foreground=request.foreground,
                background=request.background,
                font_family=(
                    request.font or "ui-monospace, SFMono-Regular, Consolas, monospace"
                ),
                output_width=request.output_width,
                output_height=request.output_height,
                fit=request.fit_mode.value,
                alignment=request.alignment_mode.value,
            )
        except (OSError, ValueError) as exc:
            raise RenderExportError(f"Could not encode SVG: {exc}") from exc
        dimensions = _graphical_dimensions(
            glyph_text,
            request.output_width,
            request.output_height,
        )
        return document, "image/svg+xml", ".svg", *dimensions
    try:
        image = render_text_png(
            glyph_text,
            foreground=request.foreground,
            background=request.background,
            font=request.font,
            output_width=request.output_width,
            output_height=request.output_height,
            fit=request.fit_mode.value,
            alignment=request.alignment_mode.value,
        )
        stream = BytesIO()
        image.save(stream, format="PNG", optimize=True)
    except (OSError, ValueError) as exc:
        raise RenderExportError(f"Could not encode PNG: {exc}") from exc
    return stream.getvalue(), "image/png", ".png", image.width, image.height


def _graphical_dimensions(
    text: str,
    output_width: int | None,
    output_height: int | None,
) -> tuple[int | None, int | None]:
    if output_width is not None and output_height is not None:
        return output_width, output_height
    lines = text.splitlines() or [""]
    intrinsic_width = max(1, *(len(line) for line in lines)) * 6.2
    intrinsic_height = max(1, len(lines)) * 10.5
    if output_width is not None:
        return output_width, max(
            1, round(output_width * intrinsic_height / intrinsic_width)
        )
    if output_height is not None:
        return max(
            1, round(output_height * intrinsic_width / intrinsic_height)
        ), output_height
    return max(1, round(intrinsic_width)), max(1, round(intrinsic_height))


def render_image(
    source: ImageSource,
    request: RenderRequest | None = None,
    *,
    destination: str | os.PathLike[str] | None = None,
) -> RenderArtifact:
    """Render one image through the maintained engine and optionally save it."""

    selected = request or RenderRequest()
    _validate_colors(selected)
    try:
        resolved_charset = AlphabetManager.resolve_alphabet(
            selected.charset,
            strict_names=True,
        )
    except ValueError as exc:
        raise RenderContractError(str(exc)) from exc
    started = perf_counter_ns()
    image = load_image(source, selected)
    if selected.optimize:
        image = ImageOps.autocontrast(image)
    loaded = perf_counter_ns()
    pixels = np.asarray(image, dtype=np.uint8)
    try:
        renderer = FrameRenderer(
            RenderConfig(
                width=selected.width,
                height=selected.height,
                mode=selected.mode,
                color=_renderer_color(selected.render_format),
                charset=resolved_charset,
                invert=selected.invert,
                dither=selected.dither,
                threshold=selected.threshold,
                edge_algorithm=selected.edge_algorithm,
                edge_threshold=selected.edge_threshold,
                cell_aspect=selected.cell_aspect,
                resample=selected.resample,
                brightness=selected.brightness,
                contrast=selected.contrast,
            )
        )
        result = renderer.render(
            pixels,
            max_width=selected.max_width,
            max_height=selected.max_height,
        )
    except RenderContractError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise RenderExecutionError(str(exc)) from exc
    rendered = perf_counter_ns()
    glyph_text = strip_ansi(result.text)
    if selected.style:
        glyph_text = apply_style(glyph_text, style_name=selected.style)
    data, media_type, suffix, pixel_width, pixel_height = _encoded_output(
        glyph_text,
        result.text,
        selected,
    )
    encoded = perf_counter_ns()
    byte_size = len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
    metrics = RenderMetrics(
        source_width=image.width,
        source_height=image.height,
        columns=result.width,
        rows=result.height,
        load_ms=(loaded - started) / 1_000_000,
        render_ms=(rendered - loaded) / 1_000_000,
        encode_ms=(encoded - rendered) / 1_000_000,
        total_ms=(encoded - started) / 1_000_000,
        output_bytes=byte_size,
    )
    artifact = RenderArtifact(
        request=selected,
        glyph_text=glyph_text,
        data=data,
        media_type=media_type,
        suffix=suffix,
        columns=result.width,
        rows=result.height,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        metrics=metrics,
    )
    if destination is not None:
        artifact.save(destination)
    return artifact


__all__ = [
    "ImageSource",
    "format_for_path",
    "load_image",
    "render_image",
    "strip_ansi",
    "truecolor_to_html",
]
