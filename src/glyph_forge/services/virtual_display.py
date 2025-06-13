"""Context manager for temporary virtual displays."""

from contextlib import contextmanager
from typing import Optional, Iterator

from .capture_virtual_display import start_virtual_display, stop_virtual_display


@contextmanager
def virtual_display(width: int = 1024, height: int = 768) -> Iterator[Optional[object]]:
    """Provide a virtual display for the duration of the context."""
    display = start_virtual_display(width=width, height=height)
    try:
        yield display
    finally:
        if display is not None:
            stop_virtual_display(display)
