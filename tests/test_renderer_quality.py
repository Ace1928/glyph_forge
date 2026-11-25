import numpy as np
import pytest

# Renderer LUT generation relies on OpenCV conversion routines.
pytest.importorskip("cv2", reason="renderer quality tests require opencv-python")

from glyph_forge.streaming.core.renderer import GlyphRenderer, RenderConfig


def test_luminance_mapping_monotonic():
    renderer = GlyphRenderer(RenderConfig())
    lum = np.arange(0, 256, dtype=np.uint8)
    idx = renderer.tables.get_char_index(lum)
    assert idx[0] <= idx[-1]
    assert np.all(np.diff(idx) >= 0)


def test_ansi256_lut_shape():
    renderer = GlyphRenderer(RenderConfig())
    lut = renderer.tables._rgb_to_ansi256
    assert lut.shape == (32, 32, 32)
    assert lut.dtype == np.uint8


def test_braille_render_dimensions():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    renderer = GlyphRenderer(RenderConfig())
    out = renderer.render_braille(frame, width=40, height=10)
    lines = out.splitlines()
    assert len(lines) == 10
    assert all(len(line) == 40 for line in lines)
