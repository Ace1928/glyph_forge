#!/usr/bin/env python3
"""Small, executable examples for the supported Glyph Forge Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

from glyph_forge import GlyphForgeError, detect_terminal, get_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a banner and optionally convert an image.",
        epilog="Example: python examples/api_examples.py photo.jpg --width 80",
    )
    parser.add_argument("image", nargs="?", type=Path, help="Optional image to convert")
    parser.add_argument("--width", type=int, default=80, help="Image width in characters")
    parser.add_argument("--color", action="store_true", help="Use adaptive terminal color")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.width < 1:
        raise SystemExit("ERROR: --width must be positive")

    forge = get_api()
    print(forge.generate_banner("GLYPH FORGE", font="slant", style="boxed"))

    if args.image is None:
        print("INFO: pass an image path to run the conversion example")
        return 0
    if not args.image.is_file():
        raise SystemExit(f"ERROR: image does not exist: {args.image}")

    color_mode = detect_terminal().preferred_color if args.color else "none"
    try:
        art = forge.image_to_glyph(
            str(args.image),
            width=args.width,
            color_mode=color_mode,
            dithering=True,
        )
    except GlyphForgeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
