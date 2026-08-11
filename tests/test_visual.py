"""Tests for the shared low-cost visual tone curve."""

from __future__ import annotations

import numpy as np
import pytest

from glyph_forge.visual import apply_tone, tone_lut
from glyph_forge.visual_defaults import DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST


def test_default_curve_is_brighter_with_clearer_midtones() -> None:
    pixels = np.asarray([0, 64, 128, 192, 255], dtype=np.uint8)
    adjusted = apply_tone(pixels)

    assert (DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST) == (1.12, 1.08)
    assert int(adjusted[0]) == 0
    assert int(adjusted[2]) > 128
    assert int(adjusted[-1]) == 255


def test_neutral_curve_is_exact_and_cached() -> None:
    pixels = np.arange(256, dtype=np.uint8)

    assert np.array_equal(apply_tone(pixels, 1.0, 1.0), pixels)
    assert tone_lut(1.25, 1.1) is tone_lut(1.25, 1.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_tone_curve_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        tone_lut(value, 1.0)
