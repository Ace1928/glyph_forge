"""Banner generation service."""

from typing import Optional, List
from ..core.banner_generator import BannerGenerator

_generator = BannerGenerator()


def text_to_banner(
    text: str,
    style: str = "minimal",
    font: Optional[str] = None,
    width: Optional[int] = None,
    effects: Optional[List[str]] = None,
    color: bool = False,
) -> str:
    """Convert text to a styled banner.

    Args:
        text: Input text to convert.
        style: Style preset name.
        font: Font name override.
        width: Output width in characters.
        effects: Optional list of special effects.
        color: Apply ANSI colors if ``True``.

    Returns:
        Glyph art banner as a string.
    """
    global _generator

    if font is not None or width is not None:
        _generator = BannerGenerator(
            font=font or _generator.font,
            width=width or _generator.width,
        )

    return _generator.generate(text, style=style, effects=effects, color=color)
