"""Directional edge-processing and CLI integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.live.edges import EdgeAlgorithm, detect_edges, directional_glyphs
from glyph_forge.live.renderers import FrameRenderer, RenderConfig, RenderMode


@pytest.mark.parametrize("algorithm", list(EdgeAlgorithm))
def test_every_edge_algorithm_preserves_shape(algorithm: EdgeAlgorithm) -> None:
    image = np.zeros((12, 18), dtype=np.uint8)
    image[:, 9:] = 255

    result = detect_edges(image, algorithm, threshold=32)

    assert result.magnitude.shape == image.shape
    assert result.gradient_x.shape == image.shape
    assert result.gradient_y.shape == image.shape
    assert result.magnitude.dtype == np.uint8
    assert result.magnitude.max() > 0


def test_vertical_boundary_maps_to_vertical_glyph() -> None:
    image = np.zeros((7, 9), dtype=np.uint8)
    image[:, 4:] = 255
    edges = detect_edges(image, "sobel", threshold=1)
    glyphs = directional_glyphs(edges)
    strongest = np.unravel_index(np.argmax(edges.magnitude), edges.magnitude.shape)

    assert glyphs[strongest] in {"│", "┃"}


def test_edge_renderer_combines_density_and_direction() -> None:
    frame = np.zeros((10, 16, 3), dtype=np.uint8)
    frame[:, 8:] = 255
    result = FrameRenderer(
        RenderConfig(
            width=16,
            height=10,
            mode="edge",
            charset=" .",
            edge_threshold=1,
        )
    ).render(frame)

    assert result.mode is RenderMode.EDGE
    assert any(glyph in result.text for glyph in "│┃")
    assert len(result.text.splitlines()) == 10


def test_image_cli_exposes_edge_renderer(tmp_path: Path) -> None:
    source = tmp_path / "edge.png"
    output = tmp_path / "edge.txt"
    pixels = np.zeros((10, 16, 3), dtype=np.uint8)
    pixels[:, 8:] = 255
    Image.fromarray(pixels).save(source)

    result = CliRunner().invoke(
        app,
        [
            "image",
            str(source),
            "--mode",
            "edge",
            "--edge-algorithm",
            "scharr",
            "--edge-threshold",
            "1",
            "--width",
            "16",
            "--height",
            "10",
            "--no-fit",
            "--no-preview",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert any(glyph in output.read_text(encoding="utf-8") for glyph in "│┃")
