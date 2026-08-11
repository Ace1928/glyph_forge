"""Focused command-layer adapter for canonical still-image rendering."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..contracts import RenderArtifact, RenderFormat, RenderRequest
from ..live.renderers import normalize_render_mode
from ..rendering import format_for_path, render_image
from ..runtime import detect_runtime_profile

_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", re.IGNORECASE)


class ImageCommandError(ValueError):
    """Invalid combination of still-image command options."""

    def __init__(self, message: str, *, param_hint: str = "image options") -> None:
        super().__init__(message)
        self.param_hint = param_hint


@dataclass(frozen=True, slots=True)
class ImageCommandOptions:
    """CLI-owned values used to build one canonical render request."""

    source: Path
    output: Path | None
    width: int | None
    height: int | None
    size: str | None
    output_width: int | None
    output_height: int | None
    fit: str
    alignment: str
    foreground: str
    background: str
    charset: str
    style: str | None
    color: str
    mode: str
    edge_algorithm: str
    edge_threshold: int
    aspect: float | None
    invert: bool
    brightness: float
    contrast: float
    optimize: bool
    dithering: bool
    fit_terminal: bool
    performance: str


@dataclass(frozen=True, slots=True)
class CharsetAction:
    """Optional list/preview response resolved before a normal image render."""

    entries: tuple[tuple[str, str], ...] = ()
    preview: str | None = None


def parse_output_size(value: str) -> tuple[int, int]:
    """Parse user-friendly WIDTHxHEIGHT or WIDTH×HEIGHT dimensions."""

    match = _SIZE_PATTERN.fullmatch(value)
    if match is None:
        raise ImageCommandError(
            "Size must use WIDTHxHEIGHT, for example 1920x1080",
            param_hint="--size",
        )
    width, height = (int(part) for part in match.groups())
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ImageCommandError(
            "Size dimensions must be between 1 and 8192 pixels",
            param_hint="--size",
        )
    return width, height


def _output_dimensions(options: ImageCommandOptions) -> tuple[int | None, int | None]:
    if options.size is None:
        return options.output_width, options.output_height
    if options.output_width is not None or options.output_height is not None:
        raise ImageCommandError(
            "Use either --size or --output-width/--output-height, not both",
            param_hint="--size",
        )
    return parse_output_size(options.size)


def _terminal_bounds() -> tuple[int, int]:
    terminal = shutil.get_terminal_size(fallback=(80, 24))
    return max(1, terminal.columns - 2), max(1, terminal.lines - 3)


def build_render_request(options: ImageCommandOptions) -> RenderRequest:
    """Validate command semantics and create the public versioned contract."""

    try:
        profile = detect_runtime_profile(options.performance)
    except ValueError as exc:
        raise ImageCommandError(str(exc), param_hint="--performance") from exc
    output_width, output_height = _output_dimensions(options)
    graphical = options.output is not None and options.output.suffix.casefold() in {
        ".png",
        ".svg",
    }
    if (output_width is not None or output_height is not None) and not graphical:
        raise ImageCommandError(
            "--size, --output-width, and --output-height require a .png or .svg output",
            param_hint="--output",
        )
    if graphical and options.color.casefold() != "none":
        raise ImageCommandError(
            "PNG/SVG exports use --foreground and --background; choose --color none",
            param_hint="--color",
        )
    try:
        output_format = format_for_path(options.output, color=options.color)
        selected_mode = normalize_render_mode(options.mode).value
    except ValueError as exc:
        hint = "--color" if "color output" in str(exc) else "--mode"
        raise ImageCommandError(str(exc), param_hint=hint) from exc
    width = options.width or profile.image_width
    height = options.height
    if options.aspect is not None and height is None:
        height = max(1, round(width / options.aspect))
    max_width: int | None = None
    max_height: int | None = None
    if options.fit_terminal and not graphical:
        max_width, max_height = _terminal_bounds()
    try:
        return RenderRequest(
            width=width,
            height=height,
            mode=selected_mode,
            output_format=output_format,
            charset=options.charset,
            invert=options.invert,
            dither=options.dithering,
            edge_algorithm=options.edge_algorithm,
            edge_threshold=options.edge_threshold,
            resample=profile.resample,
            brightness=options.brightness,
            contrast=options.contrast,
            style=options.style,
            optimize=options.optimize,
            max_width=max_width,
            max_height=max_height,
            output_width=output_width,
            output_height=output_height,
            fit=options.fit,
            alignment=options.alignment,
            foreground=options.foreground,
            background=options.background,
        )
    except ValueError as exc:
        raise ImageCommandError(str(exc)) from exc


def execute_image_command(options: ImageCommandOptions) -> RenderArtifact:
    """Render and atomically save one command request."""

    request = build_render_request(options)
    return render_image(options.source, request, destination=options.output)


def preview_payload(artifact: RenderArtifact) -> str:
    """Return the terminal-friendly representation of an artifact."""

    if artifact.request.render_format in {RenderFormat.PNG, RenderFormat.SVG}:
        return artifact.glyph_text
    if isinstance(artifact.data, bytes):
        return artifact.glyph_text
    return artifact.data


def resolve_charset_action(
    *,
    list_requested: bool,
    preview_name: str | None,
    sample_options: ImageCommandOptions | None,
) -> CharsetAction | None:
    """Resolve image-command charset discovery without touching CLI globals."""

    from ..utils.alphabet_manager import AlphabetManager

    names = sorted(AlphabetManager.list_available_alphabets())
    if list_requested:
        return CharsetAction(
            entries=tuple((name, AlphabetManager.get_alphabet(name)) for name in names)
        )
    if preview_name is None:
        if sample_options is not None:
            raise ImageCommandError(
                "--sample requires --preview-charset",
                param_hint="--sample",
            )
        return None
    if preview_name not in names:
        raise ImageCommandError(
            f"Unknown character set {preview_name!r}",
            param_hint="--preview-charset",
        )
    sections = [preview_name, AlphabetManager.get_alphabet(preview_name)]
    if sample_options is not None:
        sections.append(preview_payload(execute_image_command(sample_options)))
    return CharsetAction(preview="\n".join(sections))


__all__ = [
    "CharsetAction",
    "ImageCommandError",
    "ImageCommandOptions",
    "build_render_request",
    "execute_image_command",
    "parse_output_size",
    "preview_payload",
    "resolve_charset_action",
]
