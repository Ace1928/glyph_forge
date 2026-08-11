"""Acceptance tests for exact-size still image exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.services.image_to_glyph import ImageGlyphConverter


def _source(path: Path) -> None:
    Image.new("RGB", (80, 45), (96, 144, 208)).save(path)


def test_cli_png_has_exact_pixels_independent_from_glyph_grid(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "forge.png"
    _source(source)

    result = CliRunner().invoke(
        app,
        [
            "image",
            str(source),
            "--width",
            "24",
            "--height",
            "10",
            "--output",
            str(output),
            "--output-width",
            "333",
            "--output-height",
            "211",
            "--no-preview",
        ],
    )

    assert result.exit_code == 0, result.output
    with Image.open(output) as exported:
        assert exported.size == (333, 211)
        assert exported.format == "PNG"


def test_cli_svg_has_exact_vector_viewbox(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "forge.svg"
    _source(source)

    result = CliRunner().invoke(
        app,
        [
            "image",
            str(source),
            "--mode",
            "braille",
            "--width",
            "18",
            "--height",
            "8",
            "--output",
            str(output),
            "--output-width",
            "640",
            "--output-height",
            "360",
            "--no-preview",
        ],
    )

    assert result.exit_code == 0, result.output
    root = ET.fromstring(output.read_text(encoding="utf-8"))
    assert root.attrib["viewBox"] == "0 0 640.00 360.00"
    assert root.findall(".//{http://www.w3.org/2000/svg}text")


def test_cli_pixel_dimensions_require_a_graphical_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)

    result = CliRunner().invoke(
        app,
        ["image", str(source), "--output-width", "640", "--no-preview"],
    )

    assert result.exit_code == 2
    assert ".png or .svg" in result.output


def test_colour_stills_apply_the_same_tone_curve_as_grayscale() -> None:
    converter = ImageGlyphConverter(
        charset="@",
        width=1,
        height=1,
        brightness=2.0,
        contrast=1.0,
        auto_scale=False,
    )

    result = converter.convert_color(Image.new("RGB", (1, 1), (50, 60, 70)))

    assert "\x1b[38;2;100;120;140m" in result
