"""Utility helpers for Glyph Forge."""
from .glyph_utils import (
    sanitize_text,
    resolve_style,
    trim_margins,
    center_Glyph_art,
    measure_Glyph_art,
    detect_box_borders,
    get_terminal_size,
    detect_text_color_support,
    apply_ansi_style,
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
import time
from typing import Any, Callable, Dict

logger = logging.getLogger("glyph_forge")


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Initialize and return a package logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def configure(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder configuration hook."""
    return None


def measure_performance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Simple performance measurement decorator."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug("%s completed in %.2fms", func.__name__, duration * 1000)
        return result

    return wrapper


def detect_capabilities() -> Dict[str, Any]:
    """Return minimal capability information."""
    return {}

__all__ += [
    "setup_logger",
    "configure",
    "measure_performance",
    "detect_capabilities",
]
