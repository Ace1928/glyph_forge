"""Shared visual defaults and vectorized tone adjustment helpers.

Tone mapping is deliberately implemented as a cached 256-entry lookup table.
That gives still images, live sources, and video identical output while keeping
the per-frame cost to one indexed NumPy operation.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .visual_defaults import (
    DEFAULT_BRIGHTNESS,
    DEFAULT_CONTRAST,
    MAX_TONE,
    MIN_TONE,
)


def normalize_tone(value: float, *, name: str) -> float:
    """Validate and clamp a brightness or contrast multiplier."""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return max(MIN_TONE, min(MAX_TONE, numeric))


@lru_cache(maxsize=128)
def tone_lut(
    brightness: float = DEFAULT_BRIGHTNESS,
    contrast: float = DEFAULT_CONTRAST,
) -> NDArray[np.uint8]:
    """Return a cached RGB/luma lookup table for the requested tone curve."""

    selected_brightness = normalize_tone(brightness, name="brightness")
    selected_contrast = normalize_tone(contrast, name="contrast")
    values = np.arange(256, dtype=np.float32)
    adjusted = ((values - 127.5) * selected_contrast + 127.5) * selected_brightness
    result = np.rint(np.clip(adjusted, 0, 255)).astype(np.uint8)
    result.flags.writeable = False
    return cast(NDArray[np.uint8], result)


def apply_tone(
    pixels: NDArray[Any],
    brightness: float = DEFAULT_BRIGHTNESS,
    contrast: float = DEFAULT_CONTRAST,
) -> NDArray[np.uint8]:
    """Apply the shared tone curve to an uint8 image or sampled frame."""

    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if brightness == 1.0 and contrast == 1.0:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(tone_lut(brightness, contrast)[array])


__all__ = [
    "DEFAULT_BRIGHTNESS",
    "DEFAULT_CONTRAST",
    "MAX_TONE",
    "MIN_TONE",
    "apply_tone",
    "normalize_tone",
    "tone_lut",
]
