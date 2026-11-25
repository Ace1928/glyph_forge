"""Focused microbenchmarks for supported rendering paths."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from glyph_forge.services.image_to_glyph import ImageGlyphConverter
from glyph_forge.terminal import detect_terminal


def test_image_conversion_benchmark(benchmark) -> None:
    """Measure a representative core grayscale conversion."""

    source = Image.fromarray(
        np.random.default_rng(1928).integers(0, 256, (128, 192), dtype=np.uint8)
    )
    converter = ImageGlyphConverter(width=80, auto_scale=False)

    result = benchmark(converter.convert, source)

    assert result


def test_terminal_detection_benchmark(benchmark) -> None:
    """Keep adaptive selection inexpensive for CLI startup."""

    result = benchmark(detect_terminal)

    assert result.width > 0


def test_stream_renderer_benchmark(benchmark, tmp_path) -> None:
    """Measure a 720p BGR frame rendered to a common terminal size."""

    pytest.importorskip("cv2", reason="stream benchmark requires opencv-python")
    from glyph_forge.streaming.core.renderer import GlyphRenderer, RenderConfig

    frame = np.random.default_rng(1928).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    renderer = GlyphRenderer(RenderConfig(mode="gradient", color="ansi256", cache_dir=tmp_path))

    result = benchmark(renderer.render, frame, width=120, height=40)

    assert result
