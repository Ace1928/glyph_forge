"""Simple text-to-glyph helper."""

from .text_to_banner import text_to_banner


def text_to_glyph(text: str) -> str:
    """Render text as minimal glyph art.

    Args:
        text: Plain text input.

    Returns:
        Glyph art string using the default minimal style.
    """
    return text_to_banner(text, style="minimal")
