"""Tests for vectorized live and subpixel frame renderers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from glyph_forge.live.renderers import (
    ColorOutput,
    FrameRenderer,
    RenderConfig,
    RenderMode,
    render_svg,
)


def test_glyph_renderer_has_requested_dimensions() -> None:
    frame = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    result = FrameRenderer(RenderConfig(width=12, height=5, charset=" .#")).render(
        frame
    )

    assert result.mode is RenderMode.GLYPH
    assert result.width == 12
    assert result.height == 5
    assert [len(line) for line in result.text.splitlines()] == [12] * 5


def test_braille_uses_all_eight_subpixels() -> None:
    result = FrameRenderer(
        RenderConfig(width=1, height=1, mode="braille", threshold=128)
    ).render(np.full((4, 2, 3), 255, dtype=np.uint8))

    assert result.text == "⣿"


def test_braille_dot_mapping_matches_unicode_layout() -> None:
    frame = np.zeros((4, 2, 3), dtype=np.uint8)
    frame[0, 0] = 255

    result = FrameRenderer(RenderConfig(width=1, height=1, mode="braille")).render(
        frame
    )

    assert result.text == "⠁"


def test_quadrant_renderer_preserves_diagonal_detail() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = 255
    frame[1, 1] = 255

    result = FrameRenderer(RenderConfig(width=1, height=1, mode="quadrant")).render(
        frame
    )

    assert result.text == "▚"


def test_half_block_truecolor_keeps_independent_cell_colours() -> None:
    frame = np.asarray([[[255, 0, 0]], [[0, 0, 255]]], dtype=np.uint8)

    result = FrameRenderer(
        RenderConfig(
            width=1,
            height=1,
            mode="half-block",
            color="truecolor",
        )
    ).render(frame)

    assert "\x1b[38;2;255;0;0;48;2;0;0;255m▀" in result.text
    assert result.text.endswith("\x1b[0m")


def test_ansi256_renderer_groups_terminal_colours() -> None:
    frame = np.full((1, 3, 3), [255, 0, 0], dtype=np.uint8)

    result = FrameRenderer(
        RenderConfig(
            width=3,
            height=1,
            charset="@",
            color=ColorOutput.ANSI256,
        )
    ).render(frame)

    assert result.text.count("\x1b[38;5;196m") == 1
    assert result.text.endswith("\x1b[0m")


def test_ordered_dither_is_deterministic() -> None:
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    renderer = FrameRenderer(
        RenderConfig(width=4, height=2, mode="braille", dither=True)
    )

    assert renderer.render(frame).text == renderer.render(frame).text


def test_svg_is_valid_scalable_text_and_escapes_glyphs() -> None:
    svg = render_svg(
        np.zeros((1, 1, 3), dtype=np.uint8),
        RenderConfig(width=1, height=1, charset="&"),
    )

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "viewBox" in root.attrib
    assert "&amp;" in svg


@pytest.mark.parametrize("mode", list(RenderMode))
def test_renderers_reduce_large_sources_before_text_generation(
    mode: RenderMode,
) -> None:
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = FrameRenderer(
        RenderConfig(width=20, height=8, mode=mode, color="none")
    ).render(source)

    assert result.width == 20
    assert result.height == 8
    assert len(result.text.splitlines()) == 8


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RenderConfig(width=0), "width"),
        (RenderConfig(cell_aspect=0), "cell_aspect"),
        (RenderConfig(mode="pixels"), "render mode"),
        (RenderConfig(color="cmyk"), "color mode"),
        (RenderConfig(charset=""), "charset"),
    ],
)
def test_invalid_render_options_fail_clearly(
    config: RenderConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FrameRenderer(config)
