"""Utility helpers for Glyph Forge."""

from .glyph_utils import (
    apply_ansi_style,
    center_Glyph_art,
    detect_box_borders,
    detect_text_color_support,
    get_terminal_size,
    measure_Glyph_art,
    resolve_style,
    sanitize_text,
    trim_margins,
)

__all__ = [
    "sanitize_text",
    "resolve_style",
    "trim_margins",
    "center_Glyph_art",
    "measure_Glyph_art",
    "detect_box_borders",
    "get_terminal_size",
    "detect_text_color_support",
    "apply_ansi_style",
]
import logging
import sys
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping

if TYPE_CHECKING:
    from ..config.settings import ConfigManager

logger = logging.getLogger("glyph_forge")


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Initialize and return a package logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def configure(**sections: Mapping[str, Any]) -> "ConfigManager":
    """Persist one or more configuration sections.

    Example:
        ``configure(image={"default_width": 120}, banner={"default_font": "small"})``
    """

    from ..config.settings import get_config

    manager = get_config()
    for section, values in sections.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Configuration section {section!r} must be a mapping")
        for key, value in values.items():
            manager.set(section, str(key), value)
    return manager


def measure_performance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Simple performance measurement decorator."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug("%s completed in %.2fms", func.__name__, duration * 1000)
        return result

    return wrapper


def detect_capabilities() -> Dict[str, Any]:
    """Return terminal colour and Unicode capabilities."""

    color_level = detect_text_color_support()
    encoding = sys.stdout.encoding or ""
    return {
        "ansi16": color_level >= 1,
        "ansi256": color_level >= 2,
        "truecolor": color_level >= 3,
        "color_level": color_level,
        "unicode": "utf" in encoding.casefold(),
        "encoding": encoding or None,
    }


__all__ += [
    "setup_logger",
    "configure",
    "measure_performance",
    "detect_capabilities",
]
