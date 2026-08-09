"""Compatibility launcher for the unified ``glyph-forge image`` workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from PIL import Image, ImageOps


class OutputFormat(str, Enum):
    """Historical colour/output choices retained for import compatibility."""

    NONE = "none"
    ANSI = "ansi"
    HTML = "html"

    @classmethod
    def from_string(cls, value: str) -> "OutputFormat":
        try:
            return cls(value.casefold())
        except ValueError as exc:
            raise ValueError("Output format must be none, ansi, or html") from exc


def convert_image(
    image_path: str | Path,
    output_path: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    charset: str | None = None,
    invert: bool = False,
    color_mode: str = "none",
    dithering: bool = False,
    brightness: float | None = None,
    contrast: float | None = None,
    optimize_contrast: bool = False,
) -> str:
    """Retain the original programmatic converter on the maintained service."""

    from ..services.image_to_glyph import image_to_glyph

    source: str | Image.Image = str(image_path)
    if optimize_contrast:
        with Image.open(image_path) as image:
            source = ImageOps.autocontrast(image.convert("RGB"))
    return image_to_glyph(
        source,
        output_path=str(output_path) if output_path is not None else None,
        color_mode=color_mode,
        width=width or 100,
        height=height,
        charset=charset or "general",
        invert=invert,
        dithering=dithering,
        brightness=1.0 if brightness is None else brightness,
        contrast=1.0 if contrast is None else contrast,
    )


def preview_charset(charset: str, sample_image: str | Path | None = None) -> str:
    """Return and print a compact charset or sample-image preview."""

    from ..utils.alphabet_manager import AlphabetManager

    if charset not in AlphabetManager.list_available_alphabets():
        raise ValueError(f"Unknown character set {charset!r}")
    if sample_image is None:
        result = AlphabetManager.get_alphabet(charset)
    else:
        result = convert_image(sample_image, width=80, charset=charset)
    print(result)
    return result


_SHORT_OPTIONS = {
    "-a": "--aspect",
    "-b": "--brightness",
    "-c": "--color",
    "-d": "--dither",
    "-h": "--height",
    "-i": "--invert",
    "-s": "--charset",
}


def _translate(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    translated = []
    for value in values:
        if value == "--debug":
            continue
        translated.append(_SHORT_OPTIONS.get(value, value))
    return translated


def main(arguments: Sequence[str] | None = None) -> int:
    """Run legacy arguments through the maintained unified command."""

    from ._compat import run_unified_command

    values = sys.argv[1:] if arguments is None else list(arguments)
    if "--version" in values:
        return run_unified_command("version", [], program_name="imagize")
    return run_unified_command(
        "image",
        _translate(values),
        program_name="imagize",
    )


__all__ = ["OutputFormat", "convert_image", "main", "preview_charset"]


if __name__ == "__main__":
    raise SystemExit(main())
