"""Compatibility helpers for the unified virtual-display engine."""

from typing import Optional

from ..live.virtual import VirtualDisplayError, VirtualDisplaySession


def start_virtual_display(width: int = 1024, height: int = 768) -> Optional[object]:
    """Start an invisible virtual display.

    Args:
        width: Display width in pixels.
        height: Display height in pixels.

    Returns:
        Display object or ``None`` if pyvirtualdisplay is unavailable.
    """
    try:
        return VirtualDisplaySession(width, height).start()
    except VirtualDisplayError:
        return None


def stop_virtual_display(display: object) -> None:
    """Stop a previously started virtual display."""
    close = getattr(display, "close", None)
    if callable(close):
        close()
