"""Compatibility launcher for the unified ``glyph-forge text`` workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def create_banner(
    text: str,
    font: str = "slant",
    style: str = "minimal",
) -> str:
    """Retain the original small programmatic banner helper."""

    from ..services.text_to_banner import text_to_banner

    return text_to_banner(text, font=font, style=style)


def _translate(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    translated = []
    for value in values:
        if value == "-c":
            translated.append("--color")
        elif value == "--debug":
            continue
        else:
            translated.append(value)
    return translated


def main(arguments: Sequence[str] | None = None) -> int:
    """Run legacy arguments through the maintained unified command."""

    from ._compat import run_unified_command

    values = sys.argv[1:] if arguments is None else list(arguments)
    if "--version" in values:
        return run_unified_command("version", [], program_name="bannerize")
    return run_unified_command(
        "text",
        _translate(values),
        program_name="bannerize",
    )


__all__ = ["create_banner", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
