"""Tests for portable glyph codes (image/banner/GIF encoded as base64)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from glyph_forge.cli import app
from glyph_forge.glyphcode import (
    GlyphCodeError,
    decode_code,
    encode_auto,
    encode_banner,
    encode_gif,
    encode_image,
)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    gradient = np.zeros((64, 96, 3), dtype=np.uint8)
    gradient[..., 0] = np.arange(96, dtype=np.uint8)[None, :]
    gradient[..., 1] = np.arange(64, dtype=np.uint8)[:, None]
    gradient[..., 2] = 200
    path = tmp_path / "sample.png"
    Image.fromarray(gradient).save(path)
    return path


@pytest.fixture()
def sample_gif(tmp_path: Path, sample_image: Path) -> Path:
    path = tmp_path / "sample.gif"
    Image.open(sample_image).save(
        path,
        format="GIF",
        save_all=True,
        append_images=[
            Image.open(sample_image).rotate(90),
            Image.open(sample_image).rotate(180),
        ],
        duration=[120, 80, 200],
        loop=0,
    )
    return path


def test_image_round_trip_is_byte_exact(sample_image: Path, tmp_path: Path) -> None:
    code = encode_image(sample_image)

    assert code.startswith("glyph:v1:img:")
    assert all(ord(char) < 128 for char in code)

    decoded = decode_code(code)
    assert decoded.kind == "img"
    assert decoded.image is not None
    target = tmp_path / "out.png"
    decoded.save_image(target)
    assert target.read_bytes() == sample_image.read_bytes()


def test_gif_round_trip(sample_gif: Path, tmp_path: Path) -> None:
    code = encode_gif(sample_gif)

    assert code.startswith("glyph:v1:gif:")

    decoded = decode_code(code)
    assert decoded.is_animated
    assert len(decoded.frames) == 3
    assert decoded.frame_durations_ms == [120, 80, 200]

    target = tmp_path / "out.gif"
    decoded.save_gif(target)
    with Image.open(target) as restored:
        assert restored.n_frames == 3


def test_auto_encodes_gif_as_animation(sample_gif: Path) -> None:
    assert encode_auto(sample_gif).startswith("glyph:v1:gif:")


def test_banner_round_trip_uses_stored_style() -> None:
    code = encode_banner("GLYPH CODES", font="slant", style="boxed", width=60)

    assert code.startswith("glyph:v1:banner:")

    decoded = decode_code(code)
    assert decoded.banner is not None
    assert decoded.banner.font == "slant"
    assert decoded.banner.style == "boxed"
    banner = decoded.banner_text()
    assert banner is not None
    assert banner.lstrip().startswith("┌")
    assert "│" in banner


def test_decode_rejects_garbage() -> None:
    with pytest.raises(GlyphCodeError):
        decode_code("definitely-not-a-code")
    with pytest.raises(GlyphCodeError):
        decode_code("glyph:v1:mystery:AAAA")
    with pytest.raises(GlyphCodeError):
        decode_code("glyph:v1:img:!!!!")
    with pytest.raises(GlyphCodeError):
        decode_code("glyph:v1:gif:AAAA")


def test_size_guard_raises_on_huge_payload(sample_image: Path) -> None:
    with pytest.raises(GlyphCodeError):
        encode_image(sample_image, max_bytes=4)


def test_cli_code_and_decode_round_trip(sample_image: Path, tmp_path: Path) -> None:
    encode_result = CliRunner().invoke(app, ["link", "code", str(sample_image)])
    assert encode_result.exit_code == 0
    code = encode_result.output.strip().splitlines()[0]
    assert code.startswith("glyph:v1:")

    target = tmp_path / "restored.png"
    decode_result = CliRunner().invoke(
        app, ["link", "decode", code, "--output", str(target)]
    )
    assert decode_result.exit_code == 0, decode_result.output
    assert target.read_bytes() == sample_image.read_bytes()


def test_cli_decode_previews_image_without_output(sample_image: Path) -> None:
    code = encode_image(sample_image)
    decode_result = CliRunner().invoke(app, ["link", "decode", code])
    assert decode_result.exit_code == 0
    assert "Glyph Forge · braille" in decode_result.output


def test_cli_banner_decode_prints_art() -> None:
    code = encode_banner("HI", font="small", style="boxed")
    decode_result = CliRunner().invoke(app, ["link", "decode", code])
    assert decode_result.exit_code == 0
    assert "┌" in decode_result.output


def test_cli_decode_rejects_garbage() -> None:
    result = CliRunner().invoke(app, ["link", "decode", "nonsense"])
    assert result.exit_code == 2
    assert "not a glyph code" in result.output

    result = CliRunner().invoke(app, ["link", "decode", "glyph:v1:img:!!!!"])
    assert result.exit_code == 2


def test_cli_gif_decode_reconstructs_animation(
    sample_gif: Path, tmp_path: Path
) -> None:
    code = encode_gif(sample_gif)
    target = tmp_path / "restored.gif"
    result = CliRunner().invoke(app, ["link", "decode", code, "--output", str(target)])
    assert result.exit_code == 0
    with Image.open(target) as restored:
        assert restored.n_frames == 3


def test_banner_unicode_survives_ascii_encoding() -> None:
    code = encode_banner("héllo ✓", font="small")
    decoded = decode_code(code)
    assert decoded.banner is not None
    assert decoded.banner.text == "héllo ✓"
    banner = decoded.banner_text()
    assert banner is not None
    assert banner.strip()
