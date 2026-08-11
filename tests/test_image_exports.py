"""Acceptance tests for exact-size still image exports."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.contracts import RenderRequest
from glyph_forge.rendering import render_image
from glyph_forge.services.image_to_glyph import ImageGlyphConverter

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _source(path: Path) -> None:
    Image.new("RGB", (80, 45), (96, 144, 208)).save(path)


def test_stateful_converter_has_an_explicit_1_0_migration_warning() -> None:
    with pytest.warns(DeprecationWarning, match="RenderRequest and render_image"):
        ImageGlyphConverter()


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


def test_cli_size_shorthand_has_exact_pixels_and_atomic_output(tmp_path: Path) -> None:
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
            "--size",
            "333x211",
            "--output",
            str(output),
            "--no-preview",
        ],
    )

    assert result.exit_code == 0, result.output
    with Image.open(output) as exported:
        assert exported.size == (333, 211)
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_rejects_conflicting_pixel_size_options(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "forge.svg"
    _source(source)

    result = CliRunner().invoke(
        app,
        [
            "image",
            str(source),
            "--output",
            str(output),
            "--size",
            "640x360",
            "--output-width",
            "640",
        ],
        color=True,
    )

    assert result.exit_code == 2
    message = _ANSI_ESCAPE.sub("", result.output)
    assert "--size" in message
    assert "not both" in message


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


def test_cli_html_suffix_produces_a_safe_html_document_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "forge.html"
    _source(source)

    result = CliRunner().invoke(
        app,
        ["image", str(source), "--output", str(output), "--no-preview"],
    )

    assert result.exit_code == 0, result.output
    document = output.read_text(encoding="utf-8")
    assert document.startswith("<pre")
    assert document.endswith("</pre>")


def test_cli_and_python_contract_produce_identical_still_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "forge.txt"
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
            "--brightness",
            "1",
            "--contrast",
            "1",
            "--performance",
            "eco",
            "--no-fit",
            "--output",
            str(output),
            "--no-preview",
        ],
    )
    expected = render_image(
        source,
        RenderRequest(
            width=24,
            height=10,
            brightness=1.0,
            contrast=1.0,
            resample="bilinear",
        ),
    )

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == expected.glyph_text


def test_cli_reports_charset_typos_with_a_recovery_hint(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _source(source)

    result = CliRunner().invoke(
        app,
        ["image", str(source), "--charset", "detaled", "--no-preview"],
    )

    assert result.exit_code == 2
    assert "Unknown character set" in result.output
    assert "detailed" in result.output


def test_cli_no_longer_instantiates_the_legacy_stateful_converter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.png"
    _source(source)

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("legacy converter was constructed")

    monkeypatch.setattr(ImageGlyphConverter, "__init__", fail_if_constructed)
    result = CliRunner().invoke(
        app,
        ["image", str(source), "--width", "12", "--no-fit", "--no-preview"],
    )

    assert result.exit_code == 0, result.output


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
